//! SSH compute backend — runs experiments on a remote host over plain `ssh`.
//!
//! Phase 9.1: shells out to the system `ssh` binary, leveraging the user's
//! existing `~/.ssh/config`, agent, and key material. No extra crates pulled in.
//!
//! ```toml
//! [compute]
//! backend = "ssh"
//!
//! [compute.ssh]
//! host = "gpu-node-01"          # user@host, or any Host alias from ssh_config
//! remote_workdir = "/data/experiments"
//! identity_file = "~/.ssh/id_ed25519"   # optional
//! port = 22                                # optional
//! options = ["StrictHostKeyChecking=accept-new"]
//! ```

use crate::compute::{ComputeBackend, RawOutput};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::io::Read;
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

/// Configuration for the SSH compute backend.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SshConfig {
    /// `user@host` or any `Host` alias from `~/.ssh/config`.
    pub host: String,
    /// Working directory on the remote — `cd`'d into before the command runs.
    #[serde(default)]
    pub remote_workdir: Option<String>,
    /// Optional identity (private key) file.
    #[serde(default)]
    pub identity_file: Option<String>,
    /// Optional non-default port.
    #[serde(default)]
    pub port: Option<u16>,
    /// Extra `-o` options passed verbatim, e.g. `["StrictHostKeyChecking=accept-new"]`.
    #[serde(default)]
    pub options: Vec<String>,
}

/// Compute backend that runs experiments on a remote host over `ssh`.
#[derive(Debug, Clone)]
pub struct SshBackend {
    config: SshConfig,
}

impl SshBackend {
    /// Create a new SSH backend from a configuration block.
    pub fn new(config: SshConfig) -> anyhow::Result<Self> {
        if config.host.trim().is_empty() {
            anyhow::bail!("ssh backend: 'host' must not be empty");
        }
        Ok(Self { config })
    }

    /// Build the full argv passed to `ssh`. Exposed for unit testing.
    pub(crate) fn build_argv(&self, remote_command: &str) -> Vec<String> {
        let mut args = Vec::new();

        // Disable interactive prompts so a misconfigured host fails fast instead of hanging.
        args.push("-o".into());
        args.push("BatchMode=yes".into());

        for opt in &self.config.options {
            args.push("-o".into());
            args.push(opt.clone());
        }

        if let Some(port) = self.config.port {
            args.push("-p".into());
            args.push(port.to_string());
        }

        if let Some(identity) = &self.config.identity_file {
            args.push("-i".into());
            args.push(expand_tilde(identity));
        }

        args.push(self.config.host.clone());
        args.push(remote_command.to_string());
        args
    }

    /// Build the inner shell command run on the remote host. Exposed for unit testing.
    pub(crate) fn build_remote_command(
        &self,
        command: &str,
        env: &HashMap<String, String>,
    ) -> String {
        let mut prefix = String::new();
        if let Some(dir) = &self.config.remote_workdir {
            prefix.push_str(&format!("cd {} && ", shell_escape(dir)));
        }
        // Sort env keys so the rendered command is deterministic across runs and tests.
        let mut keys: Vec<&String> = env.keys().collect();
        keys.sort();
        for key in keys {
            let val = &env[key];
            prefix.push_str(&format!("{}={} ", key, shell_escape(val)));
        }
        format!("{}{}", prefix, command)
    }
}

impl ComputeBackend for SshBackend {
    fn submit(
        &self,
        command: &str,
        env: &HashMap<String, String>,
        timeout_secs: u64,
    ) -> anyhow::Result<RawOutput> {
        let start = Instant::now();
        let remote_command = self.build_remote_command(command, env);
        let argv = self.build_argv(&remote_command);

        let mut cmd = Command::new("ssh");
        cmd.args(&argv);
        cmd.stdout(Stdio::piped());
        cmd.stderr(Stdio::piped());

        let mut child = cmd
            .spawn()
            .map_err(|e| anyhow::anyhow!("failed to spawn ssh: {e}"))?;

        let deadline = Duration::from_secs(timeout_secs);
        let poll_interval = Duration::from_millis(100);

        loop {
            match child.try_wait() {
                Ok(Some(status)) => {
                    let duration = start.elapsed().as_secs_f64();
                    let mut stdout_buf = String::new();
                    let mut stderr_buf = String::new();
                    if let Some(mut out) = child.stdout.take() {
                        let _ = out.read_to_string(&mut stdout_buf);
                    }
                    if let Some(mut err) = child.stderr.take() {
                        let _ = err.read_to_string(&mut stderr_buf);
                    }
                    return Ok(RawOutput {
                        stdout: stdout_buf,
                        stderr: stderr_buf,
                        exit_code: status.code().unwrap_or(-1),
                        duration_secs: duration,
                    });
                }
                Ok(None) => {
                    if start.elapsed() > deadline {
                        let _ = child.kill();
                        let _ = child.wait();
                        anyhow::bail!("ssh command timed out after {timeout_secs}s (killed)");
                    }
                    std::thread::sleep(poll_interval);
                }
                Err(e) => anyhow::bail!("error waiting for ssh process: {e}"),
            }
        }
    }
}

/// Single-quote-escape a value for safe interpolation into a remote shell command.
fn shell_escape(value: &str) -> String {
    if value.is_empty() {
        return "''".into();
    }
    let safe = value
        .chars()
        .all(|c| c.is_ascii_alphanumeric() || matches!(c, '_' | '-' | '.' | '/' | ':' | '='));
    if safe {
        value.to_string()
    } else {
        // Wrap in single quotes; escape any embedded single quote as '\''
        let escaped = value.replace('\'', r"'\''");
        format!("'{escaped}'")
    }
}

/// Expand a leading `~/` to the user's home directory; otherwise return as-is.
fn expand_tilde(path: &str) -> String {
    if let Some(rest) = path.strip_prefix("~/")
        && let Some(home) = std::env::var_os("HOME")
    {
        return format!("{}/{}", home.to_string_lossy(), rest);
    }
    path.to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg() -> SshConfig {
        SshConfig {
            host: "gpu@example.com".into(),
            remote_workdir: Some("/data/exp".into()),
            identity_file: None,
            port: None,
            options: Vec::new(),
        }
    }

    #[test]
    fn rejects_empty_host() {
        let result = SshBackend::new(SshConfig {
            host: "   ".into(),
            ..Default::default()
        });
        assert!(result.is_err());
    }

    #[test]
    fn build_argv_includes_batch_mode_and_host() {
        let backend = SshBackend::new(cfg()).unwrap();
        let argv = backend.build_argv("echo hi");
        assert_eq!(argv[0], "-o");
        assert_eq!(argv[1], "BatchMode=yes");
        assert!(argv.contains(&"gpu@example.com".to_string()));
        assert_eq!(argv.last().unwrap(), "echo hi");
    }

    #[test]
    fn build_argv_includes_port_and_identity_and_options() {
        let backend = SshBackend::new(SshConfig {
            host: "host".into(),
            remote_workdir: None,
            identity_file: Some("/keys/id".into()),
            port: Some(2222),
            options: vec!["StrictHostKeyChecking=accept-new".into()],
        })
        .unwrap();
        let argv = backend.build_argv("true");
        let joined = argv.join(" ");
        assert!(joined.contains("-o BatchMode=yes"));
        assert!(joined.contains("-o StrictHostKeyChecking=accept-new"));
        assert!(joined.contains("-p 2222"));
        assert!(joined.contains("-i /keys/id"));
    }

    #[test]
    fn build_remote_command_includes_workdir_and_sorted_env() {
        let backend = SshBackend::new(cfg()).unwrap();
        let mut env = HashMap::new();
        env.insert("LR".into(), "0.001".into());
        env.insert("BS".into(), "32".into());
        let cmd = backend.build_remote_command("python train.py", &env);
        // workdir prefix
        assert!(cmd.starts_with("cd /data/exp && "));
        // env vars sorted alphabetically (BS before LR) for determinism
        let bs_pos = cmd.find("BS=").unwrap();
        let lr_pos = cmd.find("LR=").unwrap();
        assert!(bs_pos < lr_pos);
        assert!(cmd.ends_with("python train.py"));
    }

    #[test]
    fn shell_escape_passes_simple_values() {
        assert_eq!(shell_escape("0.001"), "0.001");
        assert_eq!(shell_escape("model_v2"), "model_v2");
        assert_eq!(shell_escape("/data/foo.json"), "/data/foo.json");
    }

    #[test]
    fn shell_escape_quotes_dangerous_values() {
        assert_eq!(shell_escape("hello world"), "'hello world'");
        assert_eq!(shell_escape("a;rm -rf /"), "'a;rm -rf /'");
        assert_eq!(shell_escape(""), "''");
    }

    #[test]
    fn shell_escape_handles_embedded_single_quotes() {
        // It's -> 'It'\''s'
        let out = shell_escape("It's");
        assert!(out.contains(r"'\''"));
    }

    #[test]
    fn expand_tilde_uses_home() {
        // SAFETY: this test mutates process env; it is single-threaded under cargo test
        // because we serialize HOME mutation via a fresh value and restore.
        let original = std::env::var_os("HOME");
        // SAFETY: tests in this module do not touch HOME concurrently.
        unsafe {
            std::env::set_var("HOME", "/tmp/fakehome");
        }
        assert_eq!(expand_tilde("~/foo/bar"), "/tmp/fakehome/foo/bar");
        assert_eq!(expand_tilde("/abs/path"), "/abs/path");
        // SAFETY: restoring original HOME after the test.
        unsafe {
            match original {
                Some(h) => std::env::set_var("HOME", h),
                None => std::env::remove_var("HOME"),
            }
        }
    }

    #[test]
    fn submit_fails_fast_when_ssh_binary_missing_or_host_unreachable() {
        // We don't require a live host — instead we assert that an unknown / closed host
        // returns an error within a short timeout (BatchMode=yes prevents prompts).
        let backend = SshBackend::new(SshConfig {
            host: "no-such-host-mobius-test.invalid".into(),
            options: vec!["ConnectTimeout=2".into()],
            ..Default::default()
        })
        .unwrap();
        let env = HashMap::new();
        let result = backend.submit("echo hi", &env, 10);
        // Either the ssh exit code is non-zero (couldn't resolve / connect),
        // or the submit returns Err. Both are valid "fast failure" outcomes.
        if let Ok(out) = result {
            assert_ne!(out.exit_code, 0, "expected non-zero exit, got {out:?}");
        }
    }
}

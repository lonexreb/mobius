//! Compute backend selection helper.
//!
//! Reads `[compute]` from `mobius.toml` and returns the configured
//! [`ComputeBackend`]. Currently supported: `local` (default), `ssh`.

use mobius_core::compute::{ComputeBackend, SubprocessBackend};
use mobius_core::config::MobiusConfig;
use mobius_core::ssh_backend::SshBackend;

/// Build a compute backend from the project's `mobius.toml`.
///
/// Falls back to `SubprocessBackend` when `[compute] backend` is unset or
/// equal to `"local"`. The `ssh` backend requires a populated `[compute.ssh]`
/// block.
pub fn build_backend(config: &MobiusConfig) -> anyhow::Result<Box<dyn ComputeBackend>> {
    match config.compute.backend.as_str() {
        "local" | "" => Ok(Box::new(SubprocessBackend)),
        "ssh" => {
            let ssh = config.compute.ssh.clone().ok_or_else(|| {
                anyhow::anyhow!(
                    "compute.backend = \"ssh\" requires a [compute.ssh] block in mobius.toml"
                )
            })?;
            Ok(Box::new(SshBackend::new(ssh)?))
        }
        other => anyhow::bail!("unknown compute backend '{other}' — supported: 'local', 'ssh'"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mobius_core::config::{ComputeSection, ProjectConfig};
    use mobius_core::ssh_backend::SshConfig;

    fn base_config() -> MobiusConfig {
        MobiusConfig {
            project: ProjectConfig {
                name: "t".into(),
                version: "0.1.0".into(),
            },
            experiment: Default::default(),
            bench: Default::default(),
            compute: ComputeSection::default(),
            agent: Default::default(),
        }
    }

    #[test]
    fn defaults_to_subprocess_backend() {
        let cfg = base_config();
        let backend = build_backend(&cfg).unwrap();
        // sanity: backend can be invoked on a trivial command
        let out = backend
            .submit("echo hello", &Default::default(), 5)
            .unwrap();
        assert!(out.stdout.contains("hello"));
    }

    #[test]
    fn explicit_local_works() {
        let mut cfg = base_config();
        cfg.compute.backend = "local".into();
        assert!(build_backend(&cfg).is_ok());
    }

    #[test]
    fn ssh_requires_config_block() {
        let mut cfg = base_config();
        cfg.compute.backend = "ssh".into();
        let err = match build_backend(&cfg) {
            Ok(_) => panic!("expected error"),
            Err(e) => e.to_string(),
        };
        assert!(err.contains("[compute.ssh]"));
    }

    #[test]
    fn ssh_with_config_block_constructs() {
        let mut cfg = base_config();
        cfg.compute.backend = "ssh".into();
        cfg.compute.ssh = Some(SshConfig {
            host: "user@host".into(),
            ..Default::default()
        });
        assert!(build_backend(&cfg).is_ok());
    }

    #[test]
    fn unknown_backend_errors() {
        let mut cfg = base_config();
        cfg.compute.backend = "modal".into();
        let err = match build_backend(&cfg) {
            Ok(_) => panic!("expected error"),
            Err(e) => e.to_string(),
        };
        assert!(err.contains("unknown compute backend"));
    }
}

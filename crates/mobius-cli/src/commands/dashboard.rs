//! Live TUI dashboard for monitoring Mobius experiments.
//!
//! Polls `~/.mobius/history.jsonl` and `budget.json` periodically,
//! rendering real-time metrics, experiment table, budget gauge, and
//! target progress in the terminal.

use crossterm::ExecutableCommand;
use crossterm::event::{self, Event, KeyCode, KeyEventKind};
use crossterm::terminal::{
    EnterAlternateScreen, LeaveAlternateScreen, disable_raw_mode, enable_raw_mode,
};
use mobius_core::budget::BudgetGuard;
use mobius_core::config::MobiusConfig;
use mobius_core::experiment::{ExperimentResult, ExperimentStore};
use mobius_core::store::JsonlStore;
use ratatui::prelude::*;
use ratatui::widgets::*;
use std::io::stdout;
use std::path::PathBuf;
use std::time::{Duration, Instant};

struct DashboardApp {
    store_path: PathBuf,
    budget_path: PathBuf,
    config: Option<MobiusConfig>,
    history: Vec<ExperimentResult>,
    budget: BudgetGuard,
}

impl DashboardApp {
    fn new() -> anyhow::Result<Self> {
        let mobius_dir = dirs::home_dir().unwrap_or_default().join(".mobius");
        let config = MobiusConfig::load("mobius.toml").ok();

        let store_path = mobius_dir.join("history.jsonl");
        let budget_path = mobius_dir.join("budget.json");

        let store = JsonlStore::new(&store_path)?;
        let history = store.load_all().unwrap_or_default();

        let budget = if budget_path.exists() {
            BudgetGuard::new(20.0)
                .with_state_file(&budget_path)
                .unwrap_or_else(|_| BudgetGuard::new(20.0))
        } else {
            BudgetGuard::new(20.0)
        };

        Ok(Self {
            store_path,
            budget_path,
            config,
            history,
            budget,
        })
    }

    fn poll_data(&mut self) {
        if let Ok(store) = JsonlStore::new(&self.store_path)
            && let Ok(h) = store.load_all()
        {
            self.history = h;
        }
        if self.budget_path.exists()
            && let Ok(b) =
                BudgetGuard::new(self.budget.limit).with_state_file(&self.budget_path)
        {
            self.budget = b;
        }
    }

    fn render(&self, frame: &mut Frame) {
        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Length(3), // Header
                Constraint::Length(5), // Metrics + Budget
                Constraint::Min(8),    // Experiment table
                Constraint::Length(5), // Targets
            ])
            .split(frame.area());

        self.render_header(frame, chunks[0]);
        self.render_metrics_budget(frame, chunks[1]);
        self.render_table(frame, chunks[2]);
        self.render_targets(frame, chunks[3]);
    }

    fn render_header(&self, frame: &mut Frame, area: Rect) {
        let name = self
            .config
            .as_ref()
            .map(|c| c.project.name.as_str())
            .unwrap_or("mobius");
        let title = format!(
            " MOBIUS DASHBOARD  |  {}  |  {} experiments  |  q to quit ",
            name,
            self.history.len()
        );
        let block = Block::bordered()
            .title(title)
            .border_type(BorderType::Double)
            .style(Style::default().fg(Color::Cyan));
        frame.render_widget(block, area);
    }

    fn render_metrics_budget(&self, frame: &mut Frame, area: Rect) {
        let cols = Layout::default()
            .direction(Direction::Horizontal)
            .constraints([Constraint::Percentage(60), Constraint::Percentage(40)])
            .split(area);

        // Sparkline of f1 over time
        let f1_values: Vec<u64> = self
            .history
            .iter()
            .map(|r| (r.metrics.get("f1").copied().unwrap_or(0.0) * 100.0) as u64)
            .collect();
        let latest_f1 = self
            .history
            .last()
            .and_then(|r| r.metrics.get("f1"))
            .map(|v| format!("{:.4}", v))
            .unwrap_or_else(|| "—".into());

        let sparkline = Sparkline::default()
            .block(Block::bordered().title(format!(" f1: {} ", latest_f1)))
            .data(&f1_values)
            .style(Style::default().fg(Color::Green));
        frame.render_widget(sparkline, cols[0]);

        // Budget gauge
        let ratio = if self.budget.limit > 0.0 {
            (self.budget.spent / self.budget.limit).min(1.0)
        } else {
            0.0
        };
        let gauge = Gauge::default()
            .block(Block::bordered().title(" Budget "))
            .gauge_style(Style::default().fg(if ratio > 0.8 {
                Color::Red
            } else {
                Color::Yellow
            }))
            .ratio(ratio)
            .label(format!(
                "${:.1} / ${:.1}  ({} runs left)",
                self.budget.spent,
                self.budget.limit,
                self.budget.experiments_remaining(
                    self.config
                        .as_ref()
                        .map(|c| c.experiment.cost_per_run)
                        .unwrap_or(1.0)
                )
            ));
        frame.render_widget(gauge, cols[1]);
    }

    fn render_table(&self, frame: &mut Frame, area: Rect) {
        let header = Row::new(vec!["ID", "F1", "Precision", "Recall", "Secs", "Status"]).style(
            Style::default()
                .fg(Color::Yellow)
                .add_modifier(Modifier::BOLD),
        );

        let rows: Vec<Row> = self
            .history
            .iter()
            .rev()
            .take(20)
            .map(|r| {
                Row::new(vec![
                    r.id.clone(),
                    r.metrics
                        .get("f1")
                        .map(|v| format!("{:.4}", v))
                        .unwrap_or_else(|| "—".into()),
                    r.metrics
                        .get("precision")
                        .map(|v| format!("{:.4}", v))
                        .unwrap_or_else(|| "—".into()),
                    r.metrics
                        .get("recall")
                        .map(|v| format!("{:.4}", v))
                        .unwrap_or_else(|| "—".into()),
                    format!("{:.1}", r.duration_secs),
                    format!("{:?}", r.status),
                ])
            })
            .collect();

        let table = Table::new(
            rows,
            [
                Constraint::Length(14),
                Constraint::Length(10),
                Constraint::Length(10),
                Constraint::Length(10),
                Constraint::Length(8),
                Constraint::Length(10),
            ],
        )
        .header(header)
        .block(Block::bordered().title(" Experiments "));
        frame.render_widget(table, area);
    }

    fn render_targets(&self, frame: &mut Frame, area: Rect) {
        let Some(cfg) = &self.config else {
            let block = Block::bordered().title(" Targets (no config) ");
            frame.render_widget(block, area);
            return;
        };

        let mut lines = Vec::new();
        for (metric, &target) in &cfg.experiment.targets {
            let current = self
                .history
                .iter()
                .filter_map(|r| r.metrics.get(metric).copied())
                .fold(0.0f64, f64::max);
            let ratio = (current / target).min(1.0);
            let bar_width = 20;
            let filled = (ratio * bar_width as f64) as usize;
            let bar: String = "█".repeat(filled) + &"░".repeat(bar_width - filled);
            let status = if current >= target { "MET" } else { "GAP" };
            lines.push(Line::from(format!(
                "  {:<12} {} {:.3}/{:.3}  [{}]",
                metric, bar, current, target, status
            )));
        }

        let paragraph = Paragraph::new(lines).block(Block::bordered().title(" Targets "));
        frame.render_widget(paragraph, area);
    }
}

pub fn run() -> anyhow::Result<()> {
    let mut app = DashboardApp::new()?;

    enable_raw_mode()?;
    stdout().execute(EnterAlternateScreen)?;
    let mut terminal = Terminal::new(CrosstermBackend::new(stdout()))?;

    let tick_rate = Duration::from_millis(500);
    let mut last_tick = Instant::now();

    loop {
        terminal.draw(|frame| app.render(frame))?;

        let timeout = tick_rate.saturating_sub(last_tick.elapsed());
        if event::poll(timeout)?
            && let Event::Key(key) = event::read()?
            && key.kind == KeyEventKind::Press
            && key.code == KeyCode::Char('q')
        {
            break;
        }

        if last_tick.elapsed() >= tick_rate {
            app.poll_data();
            last_tick = Instant::now();
        }
    }

    disable_raw_mode()?;
    stdout().execute(LeaveAlternateScreen)?;
    Ok(())
}

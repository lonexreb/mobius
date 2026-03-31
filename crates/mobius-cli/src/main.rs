mod commands;

use clap::{Parser, Subcommand};

#[derive(Parser)]
#[command(name = "mobius")]
#[command(about = "High-performance framework for autonomous ML experimentation")]
#[command(version)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Initialize a new Mobius project (generates mobius.toml)
    Init,
    /// Evaluate results against ground truth
    Evaluate {
        /// Path to predictions file
        #[arg(long)]
        predictions: String,
        /// Path to ground truth file
        #[arg(long)]
        ground_truth: String,
    },
    /// Show current best config and KPI gap
    Status,
    /// Show experiment history
    History {
        /// Number of recent entries to show
        #[arg(long, default_value = "10")]
        last: usize,
    },
    /// Run a single experiment
    Run {
        /// JSON config overrides
        #[arg(long, default_value = "{}")]
        config: String,
    },
    /// Get AI-powered config suggestion
    Suggest,
    /// Run parameter sweep across configs
    Sweep {
        /// JSON sweep spec, e.g. '{"learning_rate": [0.01, 0.1]}'
        #[arg(long)]
        spec: String,
    },
    /// Start autonomous agent loop
    Agent {
        /// Budget limit in USD
        #[arg(long, default_value = "20.0")]
        budget: f64,
    },
}

fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt::init();
    let cli = Cli::parse();

    match cli.command {
        Commands::Init => {
            let template = include_str!("../../../examples/mobius.toml.example");
            std::fs::write("mobius.toml", template)?;
            println!("Created mobius.toml");
        }
        Commands::Evaluate {
            predictions,
            ground_truth,
        } => commands::evaluate::run(&predictions, &ground_truth)?,
        Commands::Status => commands::status::run()?,
        Commands::History { last } => commands::history::run(last)?,
        Commands::Run { config } => commands::run::run(&config)?,
        Commands::Suggest => commands::suggest::run()?,
        Commands::Sweep { spec } => commands::sweep::run(&spec)?,
        Commands::Agent { budget } => commands::agent::run(budget)?,
    }

    Ok(())
}

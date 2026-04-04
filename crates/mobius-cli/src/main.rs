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
    /// Evaluate predictions against ground truth
    #[command(
        long_about = "Evaluate predictions against ground truth using multi-dimensional scoring.\n\n\
        Dimensions (Detection F1, Classification Accuracy, Timestamp MAE) are configured\n\
        in mobius.toml or use defaults. Results include bench score, grade, and per-dimension breakdown."
    )]
    Evaluate {
        /// Path to predictions JSON file
        #[arg(long)]
        predictions: String,
        /// Path to ground truth JSON file
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
        /// Run experiments in parallel
        #[arg(long)]
        parallel: bool,
        /// Max concurrent experiments (default: 4)
        #[arg(long, default_value = "4")]
        max_concurrency: usize,
    },
    /// Start autonomous agent loop
    #[command(long_about = "Start the autonomous experiment agent loop.\n\n\
        Iterates through ORIENT-PROPOSE-EXECUTE-EVALUATE-LEARN-DECIDE cycles\n\
        until targets are met, budget is exhausted, max iterations reached,\n\
        or a plateau is detected.\n\n\
        Requires mobius.toml in the current directory.")]
    Agent {
        /// Budget limit in USD
        #[arg(long, default_value = "20.0")]
        budget: f64,
        /// Strategy to use (gradient_guided, random, grid)
        #[arg(long, default_value = "gradient_guided")]
        strategy: String,
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
        Commands::Sweep {
            spec,
            parallel,
            max_concurrency,
        } => commands::sweep::run(&spec, parallel, max_concurrency)?,
        Commands::Agent { budget, strategy } => commands::agent::run(budget, &strategy)?,
    }

    Ok(())
}

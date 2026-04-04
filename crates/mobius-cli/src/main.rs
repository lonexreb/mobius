mod commands;
#[allow(dead_code)]
mod style;

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
    /// Live terminal dashboard showing experiment progress
    Dashboard,
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
        /// Storage backend: jsonl or sqlite
        #[arg(long, default_value = "jsonl")]
        store: String,
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
        /// Strategy to use (gradient_guided, random, grid, tpe, nsga2, ucb1)
        #[arg(long, default_value = "gradient_guided")]
        strategy: String,
        /// Enable ASHA trial pruning to stop bad experiments early
        #[arg(long)]
        pruning: bool,
        /// Storage backend: jsonl or sqlite
        #[arg(long, default_value = "jsonl")]
        store: String,
    },
    /// Show parameter importance rankings
    Importance {
        /// Primary metric to analyze
        #[arg(long, default_value = "f1")]
        metric: String,
        /// Storage backend: jsonl or sqlite
        #[arg(long, default_value = "jsonl")]
        store: String,
    },
    /// Ask for a parameter suggestion (ask-and-tell interface, outputs JSON)
    Ask {
        /// Strategy to use for suggestions
        #[arg(long, default_value = "gradient_guided")]
        strategy: String,
        /// Primary metric to optimize
        #[arg(long, default_value = "f1")]
        metric: String,
        /// Storage backend: jsonl or sqlite
        #[arg(long, default_value = "jsonl")]
        store: String,
    },
    /// Tell the system about an experiment result (ask-and-tell interface, outputs JSON)
    Tell {
        /// JSON config parameters, e.g. '{"lr": 0.01}'
        #[arg(long)]
        config: String,
        /// JSON metrics, e.g. '{"f1": 0.85}'
        #[arg(long)]
        metrics: String,
        /// Storage backend: jsonl or sqlite
        #[arg(long, default_value = "jsonl")]
        store: String,
    },
    /// Compare experiments side by side
    Compare {
        /// Experiment IDs to compare (at least 2)
        #[arg(required = true)]
        ids: Vec<String>,
        /// Storage backend: jsonl or sqlite
        #[arg(long, default_value = "jsonl")]
        store: String,
    },
    /// Enqueue a specific config for the agent to run next
    Enqueue {
        /// JSON config parameters, e.g. '{"lr": 0.01}'
        #[arg(long)]
        config: String,
        /// Storage backend: jsonl or sqlite
        #[arg(long, default_value = "jsonl")]
        store: String,
    },
    /// Export experiment history to CSV or JSON
    Export {
        /// Output format: csv or json
        #[arg(long, default_value = "csv")]
        format: String,
        /// Storage backend: jsonl or sqlite
        #[arg(long, default_value = "jsonl")]
        store: String,
        /// Output file path (default: stdout)
        #[arg(long)]
        output: Option<String>,
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
        Commands::Dashboard => commands::dashboard::run()?,
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
            store,
        } => commands::sweep::run(&spec, parallel, max_concurrency, &store)?,
        Commands::Agent {
            budget,
            strategy,
            pruning,
            store,
        } => commands::agent::run(budget, &strategy, pruning, &store)?,
        Commands::Importance { metric, store } => commands::importance::run(&metric, &store)?,
        Commands::Ask {
            strategy,
            metric,
            store,
        } => commands::ask::run(&strategy, &metric, &store)?,
        Commands::Tell {
            config,
            metrics,
            store,
        } => commands::tell::run(&config, &metrics, &store)?,
        Commands::Compare { ids, store } => commands::compare::run(&ids, &store)?,
        Commands::Enqueue { config, store } => commands::enqueue::run(&config, &store)?,
        Commands::Export {
            format,
            store,
            output,
        } => commands::export::run(&format, &store, output.as_deref())?,
    }

    Ok(())
}

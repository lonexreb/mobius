# Möbius Inference Watch

Does a draft configuration preserve every target token for your workload? Compare the reference with three draft settings on two recorded Apple M4 Max runs.

Submit a solution.json with workload (code, math, chat) and configuration (target-only, d4-p75, d4-p50, d2-p50). The identifiers encode draft lookahead and minimum probability in percent. Passing requires exact token equality on every prompt in both recordings. The metric is the smaller of the two ratios of median candidate decode tokens/s to median interleaved control decode tokens/s. The reference is normalized to 1.0. Compare configurations within one workload.

Source: Qwen2.5 7B target, 1.5B draft, llama.cpp build 7480, Metal, M4 Max 64 GB. Each domain contains eight distinct development prompts repeated across two studies. pilot.json contains aggregate counts, artifact hashes and study/run IDs. Both runs use the same prompts. This is an offline replay hill. Final mode deliberately declines to score until an independent split is available. No model download is needed to replay this catalog.

From a scratch directory with the sponsor CLI installed, run the following. Replace REPO with the absolute path to the Möbius checkout. The first command registers the local hill. Copying its contents preserves the generated version store.

    autolab hills new mobius-brief
    cp -R "$REPO/experiments/inference-watch/." .autolab/hills/mobius-brief/
    autolab hills check mobius-brief
    autolab hills commit mobius-brief -m "freeze measured pilot"
    autolab hills eval "$REPO/experiments/inference-watch/examples/baseline" -H mobius-brief -o baseline.json
    autolab hills eval "$REPO/experiments/inference-watch/examples/draft" -H mobius-brief -o draft.json

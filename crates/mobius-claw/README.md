# mobius-claw

Autonomous ML experiment agent for the [Mobius](https://github.com/lonexreb/mobius) framework.

Provides `AgentLoop` (7-step ORIENT-DECIDE cycle), strategies (`GradientGuidedTuning`, `RandomSearch`, `GridSearch`, `TpeSearch`), `LearningStore`, and hook system (`PreExecuteHook`, `PostEvaluateHook`).

namespace RekUiBridgeAgent;

internal static class SingleMotionTrialContract
{
    internal const string Schema = "rek.single_motion_trial.v1";
    internal const string AuthorityScope = "client_request_edges_only";
    internal const string AuthorityCaveat =
        "client request edge observed; server acceptance and authoritative execution are unknown";
    internal const int UnityFixedRateHz = 500;
    internal const int TrialRateHz = 50;
    internal const int FixedSubstepsPerTrialTick = UnityFixedRateHz / TrialRateHz;
    internal const int NeutralPreRollTicks = 50;
    internal const int ActionTick = NeutralPreRollTicks;
    internal const int LocomotionReleaseTick = 100;
    internal const int DurationTrialTicks = 250;
    internal const int FinalTrialTick = DurationTrialTicks - 1;
    internal const string ExpectedSha256 =
        "f00348f6f10fa706d5e48e8f31a0cbbdee1512564f819c82d7525637d68de99b";
    internal const string CanonicalJson =
        "{\"action_tick\":50,\"authority_caveat\":\"client request edge observed; server acceptance and authoritative execution are unknown\",\"authority_scope\":\"client_request_edges_only\",\"duration_ticks\":250,\"fixed_substeps_per_tick\":10,\"locomotion_release_tick\":100,\"neutral_pre_roll_ticks\":50,\"schema\":\"rek.single_motion_trial.v1\",\"selectors\":[{\"command_identity\":\"RobotInputController.VelocityCommand:[1,0,0]\",\"kind\":\"locomotion\",\"move_index\":null,\"selector\":\"forward\",\"velocity_command\":[1.0,0.0,0.0]},{\"command_identity\":\"RobotInputController.VelocityCommand:[-1,0,0]\",\"kind\":\"locomotion\",\"move_index\":null,\"selector\":\"backward\",\"velocity_command\":[-1.0,0.0,0.0]},{\"command_identity\":\"RobotInputController.VelocityCommand:[0,1,0]\",\"kind\":\"locomotion\",\"move_index\":null,\"selector\":\"strafe-left\",\"velocity_command\":[0.0,1.0,0.0]},{\"command_identity\":\"RobotInputController.VelocityCommand:[0,-1,0]\",\"kind\":\"locomotion\",\"move_index\":null,\"selector\":\"strafe-right\",\"velocity_command\":[0.0,-1.0,0.0]},{\"command_identity\":\"RobotInputController.VelocityCommand:[0,0,1]\",\"kind\":\"locomotion\",\"move_index\":null,\"selector\":\"yaw-left\",\"velocity_command\":[0.0,0.0,1.0]},{\"command_identity\":\"RobotInputController.VelocityCommand:[0,0,-1]\",\"kind\":\"locomotion\",\"move_index\":null,\"selector\":\"yaw-right\",\"velocity_command\":[0.0,0.0,-1.0]},{\"command_identity\":\"RobotInputController.ExecuteMoveByIndex:2\",\"kind\":\"move\",\"move_index\":2,\"selector\":\"move-2\",\"velocity_command\":[0.0,0.0,0.0]},{\"command_identity\":\"RobotInputController.ExecuteMoveByIndex:3\",\"kind\":\"move\",\"move_index\":3,\"selector\":\"move-3\",\"velocity_command\":[0.0,0.0,0.0]},{\"command_identity\":\"RobotInputController.ExecuteMoveByIndex:4\",\"kind\":\"move\",\"move_index\":4,\"selector\":\"move-4\",\"velocity_command\":[0.0,0.0,0.0]},{\"command_identity\":\"RobotInputController.ExecuteMoveByIndex:5\",\"kind\":\"move\",\"move_index\":5,\"selector\":\"move-5\",\"velocity_command\":[0.0,0.0,0.0]},{\"command_identity\":\"RobotInputController.ExecuteMoveByIndex:9\",\"kind\":\"move\",\"move_index\":9,\"selector\":\"move-9\",\"velocity_command\":[0.0,0.0,0.0]},{\"command_identity\":\"RobotInputController.ExecuteMoveByIndex:10\",\"kind\":\"move\",\"move_index\":10,\"selector\":\"move-10\",\"velocity_command\":[0.0,0.0,0.0]}],\"trial_rate_hz\":50,\"unity_fixed_rate_hz\":500}";

    internal static readonly SingleMotionSelector[] Selectors =
    {
        new("forward", "locomotion", 1f, 0f, 0f, null,
            "RobotInputController.VelocityCommand:[1,0,0]"),
        new("backward", "locomotion", -1f, 0f, 0f, null,
            "RobotInputController.VelocityCommand:[-1,0,0]"),
        new("strafe-left", "locomotion", 0f, 1f, 0f, null,
            "RobotInputController.VelocityCommand:[0,1,0]"),
        new("strafe-right", "locomotion", 0f, -1f, 0f, null,
            "RobotInputController.VelocityCommand:[0,-1,0]"),
        new("yaw-left", "locomotion", 0f, 0f, 1f, null,
            "RobotInputController.VelocityCommand:[0,0,1]"),
        new("yaw-right", "locomotion", 0f, 0f, -1f, null,
            "RobotInputController.VelocityCommand:[0,0,-1]"),
        new("move-2", "move", 0f, 0f, 0f, 2,
            "RobotInputController.ExecuteMoveByIndex:2"),
        new("move-3", "move", 0f, 0f, 0f, 3,
            "RobotInputController.ExecuteMoveByIndex:3"),
        new("move-4", "move", 0f, 0f, 0f, 4,
            "RobotInputController.ExecuteMoveByIndex:4"),
        new("move-5", "move", 0f, 0f, 0f, 5,
            "RobotInputController.ExecuteMoveByIndex:5"),
        new("move-9", "move", 0f, 0f, 0f, 9,
            "RobotInputController.ExecuteMoveByIndex:9"),
        new("move-10", "move", 0f, 0f, 0f, 10,
            "RobotInputController.ExecuteMoveByIndex:10"),
    };

    internal static bool TryGet(string? selector, out SingleMotionSelector value)
    {
        foreach (var candidate in Selectors)
        {
            if (!string.Equals(candidate.Selector, selector, StringComparison.Ordinal))
                continue;
            value = candidate;
            return true;
        }
        value = null!;
        return false;
    }
}

internal sealed record SingleMotionSelector(
    string Selector,
    string Kind,
    float Forward,
    float Strafe,
    float Yaw,
    int? MoveIndex,
    string CommandIdentity)
{
    internal bool IsLocomotion => MoveIndex is null;
}

namespace RekUiBridgeAgent;

internal sealed class G1HeldEventReconciler
{
    private readonly bool[] _terminalSeen =
        new bool[G1HeldInputScheduleContract.KickProbes.Length];
    private readonly bool[] _summarySeen =
        new bool[G1HeldInputScheduleContract.KickProbes.Length];

    internal void ObserveProbeEvent(string eventName, int probeOrdinal)
    {
        RequireProbeOrdinal(probeOrdinal);
        if (_summarySeen[probeOrdinal])
        {
            throw new InvalidDataException(
                $"G1 probe event {eventName} occurred after its measurement summary");
        }
    }

    internal void ObserveTerminal(int probeOrdinal)
    {
        RequireProbeOrdinal(probeOrdinal);
        if (_summarySeen[probeOrdinal])
            throw new InvalidDataException("G1 terminal lifecycle occurred after its summary");
        if (_terminalSeen[probeOrdinal])
            throw new InvalidDataException("duplicate G1 terminal lifecycle reconciliation");
        _terminalSeen[probeOrdinal] = true;
    }

    internal void ObserveSummary(int probeOrdinal, bool observationWindowComplete)
    {
        RequireProbeOrdinal(probeOrdinal);
        if (_summarySeen[probeOrdinal])
            throw new InvalidDataException("duplicate G1 summary reconciliation");
        if (observationWindowComplete && !_terminalSeen[probeOrdinal])
        {
            throw new InvalidDataException(
                "complete G1 observation summary preceded its terminal lifecycle");
        }
        _summarySeen[probeOrdinal] = true;
    }

    internal static void ValidateSendAnchor(
        G1KickProbe probe,
        int fixedSubstep,
        int scheduleTick)
    {
        if (!G1HeldInputScheduleContract.IsKickLifecycleInsideObservationWindow(
                probe,
                fixedSubstep) ||
            scheduleTick < probe.EdgeTick || scheduleTick >= probe.StopTick ||
            fixedSubstep < scheduleTick *
                G1HeldInputScheduleContract.FixedSubstepsPerScheduleTick ||
            fixedSubstep > (scheduleTick + 1) *
                G1HeldInputScheduleContract.FixedSubstepsPerScheduleTick)
        {
            throw new InvalidDataException(
                "G1 SendMoveEvent timing anchor was outside its observation window");
        }
    }

    internal bool TerminalSeen(int probeOrdinal)
    {
        RequireProbeOrdinal(probeOrdinal);
        return _terminalSeen[probeOrdinal];
    }

    internal bool SummarySeen(int probeOrdinal)
    {
        RequireProbeOrdinal(probeOrdinal);
        return _summarySeen[probeOrdinal];
    }

    private static void RequireProbeOrdinal(int probeOrdinal)
    {
        if (probeOrdinal < 0 ||
            probeOrdinal >= G1HeldInputScheduleContract.KickProbes.Length)
        {
            throw new InvalidDataException("invalid G1 reconciliation probe ordinal");
        }
    }
}

namespace RekUiBridgeAgent;

internal static class RenderedCommandMarkerContract
{
    internal const string Schema = "rek.rendered_command_marker.v1";
    internal const string RenderBinding =
        "first_post_marker_frame_is_first_rendered_frame_after_command_edge";
    internal const string Transition = "persistent_exact_rgb_rising_edge";
    internal const int RegionY = 8;
    internal const int RegionWidth = 8;
    internal const int RegionHeight = 8;
    internal const int RegionStartX = 8;
    internal const int RegionStrideX = 10;
    internal static readonly int[] PreRgb = { 0, 0, 0 };
    internal static readonly int[] PostRgb = { 255, 0, 255 };

    internal static readonly RenderedCommandMarkerSpec[] Specs =
    {
        Spec(0, 50, "walk_forward.press.1", "walk_forward:press:v1"),
        Spec(1, 150, "walk_forward.release.1", "walk_forward:release:v1"),
        Spec(2, 200, "walk_backward.press.1", "walk_backward:press:v1"),
        Spec(3, 300, "walk_backward.release.1", "walk_backward:release:v1"),
        Spec(4, 350, "strafe_left.press.1", "strafe_left:press:v1"),
        Spec(5, 450, "strafe_left.release.1", "strafe_left:release:v1"),
        Spec(6, 500, "strafe_right.press.1", "strafe_right:press:v1"),
        Spec(7, 600, "strafe_right.release.1", "strafe_right:release:v1"),
        Spec(8, 650, "yaw_left.press.1", "yaw_left:press:v1"),
        Spec(9, 750, "yaw_left.release.1", "yaw_left:release:v1"),
        Spec(10, 800, "yaw_right.press.1", "yaw_right:press:v1"),
        Spec(11, 900, "yaw_right.release.1", "yaw_right:release:v1"),
        Spec(12, 900, "move_index_2.press.1", "move_index_2:press:v1"),
        Spec(13, 1100, "move_index_3.press.1", "move_index_3:press:v1"),
        Spec(14, 1300, "move_index_4.press.1", "move_index_4:press:v1"),
        Spec(15, 1500, "move_index_5.press.1", "move_index_5:press:v1"),
        Spec(16, 1700, "move_index_9.press.1", "move_index_9:press:v1"),
        Spec(17, 1900, "move_index_10.press.1", "move_index_10:press:v1"),
        Spec(18, 2100, "walk_forward.press.2", "walk_forward:press:v1"),
        Spec(19, 2100, "move_index_2.press.2", "move_index_2:press:v1"),
        Spec(20, 2300, "walk_forward.release.2", "walk_forward:release:v1"),
        Spec(21, 2400, "walk_backward.press.2", "walk_backward:press:v1"),
        Spec(22, 2400, "move_index_3.press.2", "move_index_3:press:v1"),
        Spec(23, 2600, "walk_backward.release.2", "walk_backward:release:v1"),
    };

    private static RenderedCommandMarkerSpec Spec(
        int index,
        int scheduleTick,
        string selector,
        string commandIdentity) =>
        new(
            index,
            scheduleTick,
            selector,
            commandIdentity,
            RegionStartX + index * RegionStrideX,
            RegionY,
            RegionWidth,
            RegionHeight);
}

internal sealed record RenderedCommandMarkerSpec(
    int Index,
    int ScheduleTick,
    string Selector,
    string CommandIdentity,
    int X,
    int Y,
    int Width,
    int Height);

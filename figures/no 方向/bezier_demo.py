import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.patches import Rectangle


def cubic_bezier_formula(control_points: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Evaluate a cubic Bezier curve using the polynomial formula."""
    p0, p1, p2, p3 = control_points
    t = t[:, None]
    one_minus_t = 1.0 - t
    return (
        (one_minus_t ** 3) * p0
        + 3.0 * (one_minus_t ** 2) * t * p1
        + 3.0 * one_minus_t * (t ** 2) * p2
        + (t ** 3) * p3
    )


def de_casteljau_point(control_points: np.ndarray, t: float) -> np.ndarray:
    """Evaluate one Bezier point using De Casteljau's algorithm."""
    points = control_points.astype(float).copy()
    n = len(points)
    for r in range(1, n):
        points[: n - r] = (1.0 - t) * points[: n - r] + t * points[1 : n - r + 1]
    return points[0]


def cubic_bezier_casteljau(control_points: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Evaluate a cubic Bezier curve by De Casteljau for each sample t."""
    return np.array([de_casteljau_point(control_points, float(ti)) for ti in t])


def cubic_bezier_derivative(control_points: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Evaluate first derivative of a cubic Bezier curve."""
    p0, p1, p2, p3 = control_points
    t = t[:, None]
    one_minus_t = 1.0 - t
    return (
        3.0 * (one_minus_t ** 2) * (p1 - p0)
        + 6.0 * one_minus_t * t * (p2 - p1)
        + 3.0 * (t ** 2) * (p3 - p2)
    )


def make_pose_aligned_control_points(
    start: np.ndarray,
    end: np.ndarray,
    start_heading: np.ndarray,
    end_heading: np.ndarray,
    d1: float,
    d2: float,
) -> np.ndarray:
    """Build cubic Bezier control points from start/end pose and tangent lengths."""
    h0 = start_heading / np.linalg.norm(start_heading)
    h1 = end_heading / np.linalg.norm(end_heading)
    p0 = start
    p1 = start + d1 * h0
    p2 = end - d2 * h1
    p3 = end
    return np.vstack([p0, p1, p2, p3])


def add_vehicle(ax: plt.Axes, center: np.ndarray, length: float, width: float, color: str, label: str) -> None:
    """Draw a simplified top-view vehicle rectangle aligned with x-axis."""
    lower_left = (center[0] - length / 2.0, center[1] - width / 2.0)
    car = Rectangle(
        lower_left,
        length,
        width,
        linewidth=1.3,
        edgecolor=color,
        facecolor="none",
        linestyle="--",
        label=label,
    )
    ax.add_patch(car)


def intersect_vertical_segment(
    curve: np.ndarray,
    x_vertical: float,
    y_min: float,
    y_max: float,
    eps: float = 1e-9,
) -> list[float]:
    """Return y coordinates where a polyline curve intersects x=x_vertical within [y_min, y_max]."""
    hits: list[float] = []
    x0 = curve[:-1, 0]
    x1 = curve[1:, 0]
    y0 = curve[:-1, 1]
    y1 = curve[1:, 1]

    cross_mask = (x0 - x_vertical) * (x1 - x_vertical) <= 0.0
    indices = np.where(cross_mask)[0]

    for i in indices:
        dx = x1[i] - x0[i]
        if abs(dx) < eps:
            continue
        alpha = (x_vertical - x0[i]) / dx
        if -eps <= alpha <= 1.0 + eps:
            y_hit = y0[i] + alpha * (y1[i] - y0[i])
            if y_min - eps <= y_hit <= y_max + eps:
                hits.append(float(y_hit))

    return hits


def vehicle_corners(center: np.ndarray, heading: np.ndarray, length: float, width: float) -> np.ndarray:
    """Return 4 corners of an oriented rectangle vehicle footprint."""
    h = heading / np.linalg.norm(heading)
    n = np.array([-h[1], h[0]])
    half_l = length / 2.0
    half_w = width / 2.0
    front_center = center + half_l * h
    rear_center = center - half_l * h
    return np.array(
        [
            front_center + half_w * n,
            front_center - half_w * n,
            rear_center - half_w * n,
            rear_center + half_w * n,
        ]
    )


def cross2d(a: np.ndarray, b: np.ndarray) -> float:
    """2D cross product magnitude for vectors a and b."""
    return float(a[0] * b[1] - a[1] * b[0])


def point_in_convex_polygon(point: np.ndarray, polygon: np.ndarray, eps: float = 1e-9) -> bool:
    """Check whether a point is inside or on boundary of a convex polygon."""
    sign = 0
    n = len(polygon)
    for i in range(n):
        a = polygon[i]
        b = polygon[(i + 1) % n]
        c = cross2d(b - a, point - a)
        if abs(c) <= eps:
            continue
        current = 1 if c > 0.0 else -1
        if sign == 0:
            sign = current
        elif sign != current:
            return False
    return True


def segment_intersects_vertical_segment(
    a: np.ndarray,
    b: np.ndarray,
    x_vertical: float,
    y_min: float,
    y_max: float,
    eps: float = 1e-9,
) -> bool:
    """Check whether segment ab intersects or touches vertical segment at x=x_vertical."""
    x0, y0 = float(a[0]), float(a[1])
    x1, y1 = float(b[0]), float(b[1])

    # Collinear vertical segments.
    if abs(x0 - x_vertical) <= eps and abs(x1 - x_vertical) <= eps:
        seg_y_min = min(y0, y1)
        seg_y_max = max(y0, y1)
        return not (seg_y_max < y_min - eps or seg_y_min > y_max + eps)

    # Segment must cross the vertical line in x dimension.
    if (x0 - x_vertical) * (x1 - x_vertical) > 0.0:
        return False

    dx = x1 - x0
    if abs(dx) <= eps:
        return False

    alpha = (x_vertical - x0) / dx
    if alpha < -eps or alpha > 1.0 + eps:
        return False

    y_hit = y0 + alpha * (y1 - y0)
    return y_min - eps <= y_hit <= y_max + eps


def footprint_touches_vertical_segment(
    corners: np.ndarray,
    x_vertical: float,
    y_min: float,
    y_max: float,
    eps: float = 1e-9,
) -> bool:
    """Check whether an oriented vehicle rectangle intersects/touches a vertical boundary segment."""
    n = len(corners)

    for i in range(n):
        a = corners[i]
        b = corners[(i + 1) % n]
        if segment_intersects_vertical_segment(a, b, x_vertical, y_min, y_max, eps=eps):
            return True

    # Vertical segment endpoints inside footprint also imply overlap/touch.
    p_low = np.array([x_vertical, y_min])
    p_high = np.array([x_vertical, y_max])
    if point_in_convex_polygon(p_low, corners, eps=eps):
        return True
    if point_in_convex_polygon(p_high, corners, eps=eps):
        return True

    return False


def check_vehicle_body_boundary_contact(
    control_points: np.ndarray,
    car_length: float,
    car_width: float,
    slot_origin: np.ndarray,
    slot_length: float,
    slot_width: float,
    num_samples: int = 1200,
) -> dict[str, list[float]]:
    """Sample trajectory and report t values where body touches vertical slot boundaries."""
    t = np.linspace(0.0, 1.0, num_samples)
    centers = cubic_bezier_formula(control_points, t)
    derivatives = cubic_bezier_derivative(control_points, t)

    headings = np.zeros_like(derivatives)
    last_heading = np.array([1.0, 0.0])
    for i, vec in enumerate(derivatives):
        norm = np.linalg.norm(vec)
        if norm <= 1e-12:
            headings[i] = last_heading
        else:
            headings[i] = vec / norm
            last_heading = headings[i]

    left_x = slot_origin[0]
    right_x = slot_origin[0] + slot_length
    y_min = slot_origin[1]
    y_max = slot_origin[1] + slot_width
    contacts = {"left": [], "right": []}

    for ti, center, heading in zip(t, centers, headings):
        corners = vehicle_corners(center, heading, car_length, car_width)
        if footprint_touches_vertical_segment(corners, left_x, y_min, y_max):
            contacts["left"].append(float(ti))
        if footprint_touches_vertical_segment(corners, right_x, y_min, y_max):
            contacts["right"].append(float(ti))

    return contacts


def main() -> None:
    # Assignment parameters.
    slot_length = 6.0
    slot_width = 2.4
    car_length = 4.0
    car_width = 1.6

    # Parking slot bottom-left corner (x, y).
    slot_origin = np.array([6.0, 0.0])

    # Vehicle center path start/end. Start is fully outside the slot, end is inside and parallel.
    start_center = np.array([1.5, 3.6])
    end_center = np.array([9.0, 1.2])
    heading_start = np.array([1.0, 0.0])
    heading_end = np.array([1.0, 0.0])

    control_points = make_pose_aligned_control_points(
        start=start_center,
        end=end_center,
        start_heading=heading_start,
        end_heading=heading_end,
        d1=8.75,
        d2=1.6,
    )

    t = np.linspace(0.0, 1.0, 300)

    curve_formula = cubic_bezier_formula(control_points, t)
    curve_casteljau = cubic_bezier_casteljau(control_points, t)

    left_x = slot_origin[0]
    right_x = slot_origin[0] + slot_length
    y_low = slot_origin[1]
    y_high = slot_origin[1] + slot_width
    left_hits = intersect_vertical_segment(curve_formula, left_x, y_low, y_high)
    right_hits = intersect_vertical_segment(curve_formula, right_x, y_low, y_high)

    body_contacts = check_vehicle_body_boundary_contact(
        control_points=control_points,
        car_length=car_length,
        car_width=car_width,
        slot_origin=slot_origin,
        slot_length=slot_length,
        slot_width=slot_width,
    )

    max_diff = np.max(np.linalg.norm(curve_formula - curve_casteljau, axis=1))
    print(f"Max pointwise difference (formula vs Casteljau): {max_diff:.3e}")

    fig, ax = plt.subplots(figsize=(10, 6))

    # Draw road baseline and parking slot.
    ax.plot([0.0, 13.0], [0.0, 0.0], color="tab:blue", linewidth=1.6)
    slot_rect = Rectangle(
        tuple(slot_origin),
        slot_length,
        slot_width,
        linewidth=1.8,
        edgecolor="tab:red",
        facecolor="none",
        label="Parking slot (6.0m x 2.4m)",
    )
    ax.add_patch(slot_rect)

    ax.plot(
        control_points[:, 0],
        control_points[:, 1],
        "o--",
        linewidth=1.3,
        color="tab:gray",
        label="Control polygon",
    )
    ax.plot(
        curve_formula[:, 0],
        curve_formula[:, 1],
        linewidth=2.0,
        color="tab:blue",
        label="Cubic Bezier (formula)",
    )
    ax.plot(
        curve_casteljau[:, 0],
        curve_casteljau[:, 1],
        linestyle=":",
        linewidth=2.0,
        color="tab:orange",
        label="Cubic Bezier (De Casteljau)",
    )

    for i, (x, y) in enumerate(control_points):
        ax.text(x + 0.06, y + 0.06, f"P{i}")

    add_vehicle(
        ax,
        center=start_center,
        length=car_length,
        width=car_width,
        color="tab:green",
        label="Car start (4.0m x 1.6m)",
    )
    add_vehicle(
        ax,
        center=end_center,
        length=car_length,
        width=car_width,
        color="tab:purple",
        label="Car final (parallel to slot edge)",
    )

    ax.text(slot_origin[0] + slot_length / 2.0, -0.22, "6.0 m", ha="center", va="top", fontsize=12)
    ax.text(
        slot_origin[0] + slot_length + 0.35,
        slot_origin[1] + slot_width / 2.0,
        "2.4 m",
        ha="left",
        va="center",
        fontsize=12,
    )

    ax.set_title("Cubic Bezier Side-Parking Path Planning")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.25)
    ax.set_xlim(0.0, 13.0)
    ax.set_ylim(-0.6, 5.0)
    ax.legend(loc="upper right")

    print(f"Slot size: {slot_length:.1f}m x {slot_width:.1f}m")
    print(f"Car size: {car_length:.1f}m x {car_width:.1f}m")
    print(f"Start center: {start_center}, End center: {end_center}")
    print(f"Left boundary intersections: {len(left_hits)}")
    print(f"Right boundary intersections: {len(right_hits)}")
    print(f"Body contacts on left boundary: {len(body_contacts['left'])}")
    print(f"Body contacts on right boundary: {len(body_contacts['right'])}")

    if left_hits or right_hits:
        print("Warning: curve intersects vertical slot boundaries.")
    else:
        print("Constraint satisfied: no intersections with vertical slot boundaries.")

    if body_contacts["left"] or body_contacts["right"]:
        print("Warning: vehicle body touches vertical slot boundaries during motion.")
    else:
        print("Constraint satisfied: vehicle body keeps clear of vertical slot boundaries.")

    output_dir = Path(__file__).resolve().parent / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "bezier_side_parking_cubic.png"
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    print(f"Saved figure: {output_path}")

    plt.show()


if __name__ == "__main__":
    main()

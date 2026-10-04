import type { RawRle } from "../types";
import { bitmapToRle, rleArea, rleToBitmap } from "./rle";
import type { FramePoint } from "./raster";

/**
 * Vector outline of a binary mask.
 *
 * Masks are stored as RLE bitmaps, so an editable polygon has to be derived from
 * the pixels. `traceRings` walks the crack edges around every foreground blob and
 * every enclosed hole (rectilinear, exact), `simplifyRings` strips the staircase
 * down to a handful of draggable vertices, and `ringsToRle` bakes the result back
 * into the RLE the rest of the app speaks.
 *
 * A ring is stored **open**: the last vertex connects back to the first, and the
 * first vertex is not repeated at the end.
 *
 * Everything here is pure and DOM-free, so it can be exercised outside a browser.
 */

/** Douglas–Peucker tolerance, in frame pixels, the outline starts at. */
const MIN_EPSILON = 1.2;
/** Simplifying stops once the outline is at or below this many vertices. */
const MAX_VERTICES = 400;
/** A blob or hole smaller than this many pixels is not worth a ring. */
const MIN_RING_AREA = 4;
/** Ceiling for the epsilon search, so a pathological mask cannot spin forever. */
const MAX_EPSILON = 64;
/** Belt and braces against a malformed edge set. */
const MAX_STEPS = 1 << 22;

/** Step per heading, indexed 0 right / 1 down / 2 left / 3 up (screen coords). */
const STEP_X = [1, 0, -1, 0];
const STEP_Y = [0, 1, 0, -1];
/**
 * Turn preference when a corner has more than one successor (two blobs meeting
 * diagonally): right, straight, left, back. Turning right first keeps the walk
 * hugging the foreground, and the diagonal pair comes out as two rings that meet
 * at the corner rather than one self-touching ring.
 */
const TURN_ORDER = [1, 0, 3, 2];

export interface OutlineVertexHit {
    ring: number;
    index: number;
    distance: number;
}

export interface OutlineEdgeHit {
    ring: number;
    /** The edge from vertex `index` to vertex `index + 1`. */
    index: number;
    /** The projection of the query point onto that edge. */
    at: FramePoint;
    distance: number;
}

/** The mask as a bitmap indexed `x * height + y` (the app's column-major order). */
function gridOf(rle: RawRle): {
    data: Uint8Array;
    width: number;
    height: number;
} {
    const [height, width] = rle.size;
    return { data: rleToBitmap(rle), width, height };
}

function headingOf(dx: number, dy: number): number {
    for (let i = 0; i < 4; i++) {
        if (STEP_X[i] === dx && STEP_Y[i] === dy) return i;
    }
    return -1;
}

/**
 * Every blob boundary and every hole boundary, as closed rectilinear rings.
 *
 * Each foreground pixel contributes a directed crack edge for every side facing
 * background, and the edges are then linked corner to corner on the
 * `(width + 1) × (height + 1)` grid of pixel boundaries. Because RLE counts are
 * column-major, the bitmap is too, which the `x * height + y` index mirrors.
 */
export function traceRings(rle: RawRle): FramePoint[][] {
    const { data, width, height } = gridOf(rle);
    if (width === 0 || height === 0) return [];
    const at = (x: number, y: number): number =>
        x >= 0 && y >= 0 && x < width && y < height ? data[x * height + y] : 0;

    const stride = height + 1;
    /** corner index → the corners each outgoing edge points at (-1 = used). */
    const edges = new Map<number, number[]>();
    const addEdge = (ax: number, ay: number, bx: number, by: number): void => {
        const from = ax * stride + ay;
        const to = bx * stride + by;
        const list = edges.get(from);
        if (list) list.push(to);
        else edges.set(from, [to]);
    };

    for (let x = 0; x < width; x++) {
        for (let y = 0; y < height; y++) {
            if (!at(x, y)) continue;
            // Clockwise in screen coordinates, so the foreground stays inside and
            // the turns below are all right turns.
            if (!at(x, y - 1)) addEdge(x, y, x + 1, y);
            if (!at(x + 1, y)) addEdge(x + 1, y, x + 1, y + 1);
            if (!at(x, y + 1)) addEdge(x + 1, y + 1, x, y + 1);
            if (!at(x - 1, y)) addEdge(x, y + 1, x, y);
        }
    }

    const rings: FramePoint[][] = [];
    // A Map iterates in insertion order, so the result is deterministic.
    for (const [start, list] of edges) {
        for (let slot = 0; slot < list.length; slot++) {
            if (list[slot] < 0) continue;
            const ring: FramePoint[] = [];
            let corner = start;
            let heading = -1;
            let closed = false;
            for (let guard = 0; guard < MAX_STEPS; guard++) {
                const cx = Math.floor(corner / stride);
                const cy = corner - cx * stride;
                ring.push({ x: cx, y: cy });
                const outgoing = edges.get(corner);
                if (!outgoing) break;
                let choice = -1;
                let choiceHeading = -1;
                let bestRank = TURN_ORDER.length;
                for (let i = 0; i < outgoing.length; i++) {
                    const to = outgoing[i];
                    if (to < 0) continue;
                    const nx = Math.floor(to / stride);
                    const ny = to - nx * stride;
                    const dir = headingOf(nx - cx, ny - cy);
                    if (dir < 0) continue;
                    const rank =
                        heading < 0
                            ? 0
                            : TURN_ORDER.indexOf((dir - heading + 4) % 4);
                    if (rank < bestRank) {
                        bestRank = rank;
                        choice = i;
                        choiceHeading = dir;
                    }
                }
                if (choice < 0) break;
                const next = outgoing[choice];
                outgoing[choice] = -1;
                heading = choiceHeading;
                corner = next;
                if (corner === start) {
                    closed = true;
                    break;
                }
            }
            // Fewer than three corners cannot enclose anything.
            if (closed && ring.length >= 3) rings.push(ring);
        }
    }
    return rings;
}

/** Twice-signed area, used to size a ring without a bitmap. */
function signedArea(points: FramePoint[]): number {
    let sum = 0;
    for (let i = 0; i < points.length; i++) {
        const a = points[i];
        const b = points[(i + 1) % points.length];
        sum += a.x * b.y - b.x * a.y;
    }
    return sum / 2;
}

function distanceToSegment(
    point: FramePoint,
    a: FramePoint,
    b: FramePoint,
): number {
    const dx = b.x - a.x;
    const dy = b.y - a.y;
    const lengthSquared = dx * dx + dy * dy;
    if (lengthSquared === 0) return Math.hypot(point.x - a.x, point.y - a.y);
    let t = ((point.x - a.x) * dx + (point.y - a.y) * dy) / lengthSquared;
    t = Math.max(0, Math.min(1, t));
    return Math.hypot(point.x - (a.x + t * dx), point.y - (a.y + t * dy));
}

/** Ramer–Douglas–Peucker on an open polyline, keeping both ends. */
function simplifyOpen(points: FramePoint[], epsilon: number): FramePoint[] {
    if (points.length <= 2) return [...points];
    const keep = new Uint8Array(points.length);
    keep[0] = 1;
    keep[points.length - 1] = 1;
    const stack: [number, number][] = [[0, points.length - 1]];
    while (stack.length > 0) {
        const [first, last] = stack.pop() as [number, number];
        let index = -1;
        let worst = epsilon;
        for (let i = first + 1; i < last; i++) {
            const d = distanceToSegment(points[i], points[first], points[last]);
            if (d > worst) {
                worst = d;
                index = i;
            }
        }
        if (index !== -1) {
            keep[index] = 1;
            stack.push([first, index], [index, last]);
        }
    }
    return points.filter((_, i) => keep[i] === 1);
}

/**
 * Drop vertices that only continue a straight edge.
 *
 * The trace anchors a vertex at the split point it simplified around, and the
 * wrapping edge back to vertex 0 is invisible to RDP, so a collinear last vertex
 * survives it. This closes that gap: a vertex is kept only when it actually bends
 * the ring.
 */
function dropCollinear(points: FramePoint[], epsilon: number): FramePoint[] {
    const count = points.length;
    if (count <= 3) return [...points];
    const kept: FramePoint[] = [];
    for (let i = 0; i < count; i++) {
        const previous = points[(i - 1 + count) % count];
        const next = points[(i + 1) % count];
        if (distanceToSegment(points[i], previous, next) > epsilon)
            kept.push(points[i]);
    }
    return kept.length >= 3 ? kept : [...points];
}

/** Simplify a closed ring by splitting it at its farthest point from vertex 0. */
export function simplifyRing(
    points: FramePoint[],
    epsilon: number,
): FramePoint[] {
    if (points.length <= 3) return [...points];
    const first = points[0];
    let far = 1;
    let best = -1;
    for (let i = 1; i < points.length; i++) {
        const d = (points[i].x - first.x) ** 2 + (points[i].y - first.y) ** 2;
        if (d > best) {
            best = d;
            far = i;
        }
    }
    const head = simplifyOpen(points.slice(0, far + 1), epsilon);
    const tail = simplifyOpen(points.slice(far), epsilon);
    return dropCollinear([...head, ...tail.slice(1)], epsilon);
}

export function simplifyRings(
    rings: FramePoint[][],
    epsilon: number,
): FramePoint[][] {
    return rings.map((ring) => simplifyRing(ring, epsilon));
}

export function countVertices(rings: FramePoint[][]): number {
    let total = 0;
    for (const ring of rings) total += ring.length;
    return total;
}

/**
 * The editable outline of a mask: every meaningful blob and hole, simplified
 * until the vertex count is workable.
 *
 * The tolerance rises until the count fits because a brush or model mask is
 * staircased — at a pixel of tolerance a 1080p mask can still carry thousands of
 * vertices, which is not a polygon anyone can drag.
 */
export function outlineFromRle(
    rle: RawRle,
    maxVertices = MAX_VERTICES,
): FramePoint[][] {
    const traced = traceRings(rle).filter(
        (ring) => Math.abs(signedArea(ring)) >= MIN_RING_AREA,
    );
    if (traced.length === 0) return [];
    let epsilon = MIN_EPSILON;
    let simplified = simplifyRings(traced, epsilon);
    while (countVertices(simplified) > maxVertices && epsilon < MAX_EPSILON) {
        epsilon *= 1.4;
        simplified = simplifyRings(traced, epsilon);
    }
    return simplified.filter((ring) => ring.length >= 3);
}

/**
 * Rasterise rings back to RLE, with an even-odd fill so a hole (a ring inside
 * another) subtracts. Sampling at pixel centres reproduces a traced-and-unsimplified
 * mask exactly, which is why merely entering the outline editor changes nothing.
 */
export function ringsToRle(
    rings: FramePoint[][],
    width: number,
    height: number,
): RawRle | null {
    const data = new Uint8Array(width * height);
    const crossings: number[] = [];
    for (let x = 0; x < width; x++) {
        const px = x + 0.5;
        crossings.length = 0;
        for (const ring of rings) {
            for (let i = 0; i < ring.length; i++) {
                const a = ring[i];
                const b = ring[(i + 1) % ring.length];
                if (a.x === b.x) continue;
                const minX = Math.min(a.x, b.x);
                const maxX = Math.max(a.x, b.x);
                if (px < minX || px >= maxX) continue;
                const t = (px - a.x) / (b.x - a.x);
                crossings.push(a.y + t * (b.y - a.y));
            }
        }
        if (crossings.length < 2) continue;
        crossings.sort((left, right) => left - right);
        for (let i = 0; i + 1 < crossings.length; i += 2) {
            const from = Math.max(0, Math.ceil(crossings[i] - 0.5));
            const to = Math.min(
                height - 1,
                Math.ceil(crossings[i + 1] - 0.5) - 1,
            );
            for (let y = from; y <= to; y++) data[x * height + y] = 1;
        }
    }
    const rle = bitmapToRle(data, height, width);
    return rleArea(rle) === 0 ? null : rle;
}

/** The vertex closest to `point`, within `maxDistance`. */
export function nearestVertex(
    rings: FramePoint[][],
    point: FramePoint,
    maxDistance: number,
): OutlineVertexHit | null {
    let best: OutlineVertexHit | null = null;
    for (let ring = 0; ring < rings.length; ring++) {
        const points = rings[ring];
        for (let index = 0; index < points.length; index++) {
            const distance = Math.hypot(
                points[index].x - point.x,
                points[index].y - point.y,
            );
            if (distance <= maxDistance && (!best || distance < best.distance))
                best = { ring, index, distance };
        }
    }
    return best;
}

/** The closest point on any ring edge, within `maxDistance`. */
export function nearestEdge(
    rings: FramePoint[][],
    point: FramePoint,
    maxDistance: number,
): OutlineEdgeHit | null {
    let best: OutlineEdgeHit | null = null;
    for (let ring = 0; ring < rings.length; ring++) {
        const points = rings[ring];
        for (let index = 0; index < points.length; index++) {
            const a = points[index];
            const b = points[(index + 1) % points.length];
            const distance = distanceToSegment(point, a, b);
            if (distance > maxDistance || (best && distance >= best.distance))
                continue;
            const dx = b.x - a.x;
            const dy = b.y - a.y;
            const lengthSquared = dx * dx + dy * dy;
            const t =
                lengthSquared === 0
                    ? 0
                    : Math.max(
                          0,
                          Math.min(
                              1,
                              ((point.x - a.x) * dx + (point.y - a.y) * dy) /
                                  lengthSquared,
                          ),
                      );
            best = {
                ring,
                index,
                at: { x: a.x + t * dx, y: a.y + t * dy },
                distance,
            };
        }
    }
    return best;
}

/** A copy of `rings` with one vertex moved. */
export function moveVertex(
    rings: FramePoint[][],
    ring: number,
    index: number,
    to: FramePoint,
): FramePoint[][] {
    return rings.map((points, r) =>
        r === ring
            ? points.map((point, i) => (i === index ? { ...to } : point))
            : points,
    );
}

/** A copy of `rings` with `at` inserted after vertex `index` on `ring`. */
export function insertVertex(
    rings: FramePoint[][],
    ring: number,
    index: number,
    at: FramePoint,
): FramePoint[][] {
    return rings.map((points, r) => {
        if (r !== ring) return points;
        const next = [...points];
        next.splice(index + 1, 0, { ...at });
        return next;
    });
}

/**
 * A copy of `rings` with one vertex dropped, or `null` when the ring would be
 * left with fewer than three vertices (which is not a shape any more).
 */
export function removeVertex(
    rings: FramePoint[][],
    ring: number,
    index: number,
): FramePoint[][] | null {
    if ((rings[ring]?.length ?? 0) <= 3) return null;
    return rings.map((points, r) =>
        r === ring ? points.filter((_, i) => i !== index) : points,
    );
}

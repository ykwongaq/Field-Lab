import assert from "node:assert/strict";
import { test } from "node:test";
import { createServer } from "vite";
import { removedTimelineFrames } from "../src/lib/timelineClipEdit.ts";

const clip = {
    originalFirst: 0,
    originalLast: 59,
    first: 0,
    last: 59,
    deleted: false,
    removeRange: null,
};

test("a middle range removes exactly the inclusive frames, leaving both sides", () => {
    assert.deepEqual(
        removedTimelineFrames({ ...clip, removeRange: { first: 20, last: 29 } }),
        [20, 21, 22, 23, 24, 25, 26, 27, 28, 29],
    );
    assert.deepEqual(removedTimelineFrames({ ...clip, removeRange: { first: 30, last: 30 } }), [30]);
});

test("edge trims and a selected range combine without duplicate removals", () => {
    assert.deepEqual(
        removedTimelineFrames({
            ...clip,
            first: 2,
            last: 57,
            removeRange: { first: 20, last: 21 },
        }),
        [0, 1, 20, 21, 58, 59],
    );
    assert.deepEqual(removedTimelineFrames({ ...clip, removeRange: { first: 0, last: 2 } }), [0, 1, 2]);
    assert.deepEqual(removedTimelineFrames({ ...clip, removeRange: { first: 58, last: 59 } }), [58, 59]);
});

test("delete and full-range removal remove the whole clip; unchanged removes nothing", () => {
    const all = Array.from({ length: 60 }, (_, frame) => frame);
    assert.deepEqual(removedTimelineFrames({ ...clip, deleted: true }), all);
    assert.deepEqual(removedTimelineFrames({ ...clip, removeRange: { first: 0, last: 59 } }), all);
    assert.deepEqual(removedTimelineFrames(clip), []);
});

test("applying a middle removal preserves the tracklet, other masks and dirty frames", async () => {
    const server = await createServer({
        configFile: false,
        cacheDir: new URL("../node_modules/.vite/timeline-clip-test", import.meta.url).pathname,
        server: { middlewareMode: true },
    });
    try {
        const { Clip } = await server.ssrLoadModule(
            new URL("../src/lib/clip.ts", import.meta.url).pathname,
        );
        for (const mode of ["instance", "semantic"]) {
            const mask = { size: [1, 1], counts: [0, 1] };
            const dataset = {
                videos: [{ id: 1, width: 1, height: 1, segmentation_mode: mode, file_names: Array.from({ length: 60 }, (_, i) => `${i}.png`) }],
                annotations: [1, 2].map((id) => ({
                    id, object_id: id, video_id: 1, category_id: id,
                    segmentations: Array.from({ length: 60 }, () => mask),
                })),
                categories: [{ id: 1, name: "first" }, { id: 2, name: "second" }],
            };
            const archive = {
                hasEntry: (name) => name === "annotation.json",
                readAsText: async () => JSON.stringify(dataset),
            };
            const before = await Clip.fromZip(archive);
            let after = before;
            for (const frame of removedTimelineFrames({ ...clip, removeRange: { first: 20, last: 29 } }))
                after = after.removeMask(1, frame);

            assert.equal(after.tracklets.length, 2, mode);
            assert.equal(after.tracklets[0].id, 1, mode);
            assert.equal(after.tracklets[0].maskFrames.count, 50, mode);
            assert.equal(after.tracklets[1].maskFrames.count, 60, mode);
            for (let frame = 0; frame < 60; frame++) {
                assert.equal(after.rawMaskAt(after.tracklets[0], frame) === null, frame >= 20 && frame <= 29, `${mode} frame ${frame}`);
                assert.notEqual(after.rawMaskAt(after.tracklets[1], frame), null, `${mode} other tracklet frame ${frame}`);
                assert.equal(after.dirtyFrames.has(frame), frame >= 20 && frame <= 29, `${mode} dirty frame ${frame}`);
            }
            assert.equal(before.tracklets[0].maskFrames.count, 60, mode);
        }
    } finally {
        await server.close();
    }
});

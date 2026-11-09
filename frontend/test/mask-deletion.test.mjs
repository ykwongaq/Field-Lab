import assert from "node:assert/strict";
import { test } from "node:test";
import { createServer } from "vite";

test("frame deletion survives save and reopen and can be redrawn or merged", async () => {
    const server = await createServer({
        configFile: false,
        cacheDir: new URL("../node_modules/.vite/mask-deletion-test", import.meta.url).pathname,
        server: { middlewareMode: true, hmr: false },
    });
    try {
        const { Clip } = await server.ssrLoadModule(
            new URL("../src/lib/clip.ts", import.meta.url).pathname,
        );
        const { ZipArchive } = await server.ssrLoadModule(
            new URL("../src/lib/zip.ts", import.meta.url).pathname,
        );
        const { ZipWriter } = await server.ssrLoadModule(
            new URL("../src/lib/zipWriter.ts", import.meta.url).pathname,
        );
        const { planTrackletMerge } = await server.ssrLoadModule(
            new URL("../src/lib/merge.ts", import.meta.url).pathname,
        );
        const mask = { size: [1, 1], counts: [0, 1] };
        const dataset = {
            videos: [{ id: 1, width: 1, height: 1, file_names: ["0.png", "1.png", "2.png"] }],
            annotations: [
                { id: 1, object_id: 1, video_id: 1, category_id: 1, segmentations: [mask, mask, null] },
                { id: 2, object_id: 2, video_id: 1, category_id: 2, segmentations: [null, null, mask] },
            ],
            categories: [{ id: 1, name: "first" }, { id: 2, name: "second" }],
        };
        const writer = new ZipWriter();
        writer.addText("annotation.json", JSON.stringify(dataset));
        const archive = await ZipArchive.fromFile(new File([writer.toBlob()], "test.project"));
        const before = await Clip.fromZip(archive);
        const deleted = before.removeMask(1, 0);

        assert.equal(deleted.editCount, before.editCount + 1);
        assert.equal(deleted.dirtyFrames.has(0), true);
        assert.equal(deleted.rawMaskAt(deleted.tracklets[0], 0), null);
        assert.notEqual(deleted.rawMaskAt(deleted.tracklets[0], 1), null);
        assert.notEqual(before.rawMaskAt(before.tracklets[0], 0), null);

        const saved = await deleted.exportProjectZip(archive);
        const reopenedArchive = await ZipArchive.fromFile(new File([saved], "saved.project"));
        const reopened = await Clip.fromZip(reopenedArchive);
        assert.equal(reopened.rawMaskAt(reopened.tracklets[0], 0), null);
        assert.notEqual(reopened.rawMaskAt(reopened.tracklets[0], 1), null);

        const redrawn = reopened.replaceMask(1, 0, mask);
        assert.notEqual(redrawn.rawMaskAt(redrawn.tracklets[0], 0), null);
        assert.equal(redrawn.tracklets.length, 2);

        const target = reopened.tracklets[0];
        const plan = planTrackletMerge({
            target,
            sources: [reopened.tracklets[1]],
            scope: { kind: "all" },
            policy: "union",
            labelId: target.labelId,
            frameCount: reopened.frameCount,
            isHuman: () => false,
        });
        const merged = reopened.applyMerge(plan);
        assert.equal(merged.tracklets.length, 1);
        assert.equal(merged.rawMaskAt(merged.tracklets[0], 0), null);
        assert.notEqual(merged.rawMaskAt(merged.tracklets[0], 1), null);
        assert.notEqual(merged.rawMaskAt(merged.tracklets[0], 2), null);
        assert.equal(reopened.removeMask(1, 1).tracklets.some((t) => t.id === 1), false);
    } finally {
        await server.close();
    }
});

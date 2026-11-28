import assert from "node:assert/strict";
import { after, test } from "node:test";
import { createServer } from "vite";

// Run the actual key handler without a browser by substituting the effect scheduler.
const server = await createServer({
    configFile: false,
    server: { middlewareMode: true },
    ssr: { noExternal: ["react"] },
    plugins: [{
        name: "test-shortcut-effect",
        enforce: "pre",
        resolveId(id, importer) {
            if (id === "react" && importer?.includes("useWorkspaceShortcuts"))
                return "\0test-shortcut-react";
        },
        load(id) {
            if (id === "\0test-shortcut-react")
                return "export function useEffect(effect) { effect(); }";
        },
    }],
});
after(() => server.close());

const { useWorkspaceShortcuts: registerShortcuts } = await server.ssrLoadModule(
    new URL("../src/hooks/useWorkspaceShortcuts.ts", import.meta.url).pathname,
);

function installShortcuts(overrides = {}) {
    let onKeyDown;
    const steps = [];
    const tools = [];
    globalThis.window = {
        addEventListener(type, handler) {
            assert.equal(type, "keydown");
            onKeyDown = handler;
        },
        removeEventListener() {},
    };
    registerShortcuts({
        exportOpen: false,
        tool: "review",
        method: "point",
        polygonLength: 0,
        hasPropRun: false,
        propagating: false,
        ...overrides,
    }, {
        stepFrame: (delta) => steps.push(delta),
        changeTool: (tool) => tools.push(tool),
        changeMethod() {},
        closePolygon() {},
        commitMask() {},
        undo() {},
        acceptPropagation() {},
        async runPropagation() {},
        setPlaying() {},
        setBrushSize() {},
    });
    return {
        steps,
        tools,
        press(key, options = {}) {
            const event = {
                key,
                target: { tagName: "DIV" },
                defaultPrevented: false,
                preventDefault() { this.defaultPrevented = true; },
                ...options,
            };
            onKeyDown(event);
            return event;
        },
    };
}

test("A and D step frames in each tool, alongside the arrow keys", () => {
    for (const tool of ["review", "addMask", "editMask", "propagate"]) {
        const shortcuts = installShortcuts({ tool });
        for (const key of ["a", "d", "ArrowLeft", "ArrowRight"])
            assert.equal(shortcuts.press(key).defaultPrevented, true);
        assert.deepEqual(shortcuts.steps, [-1, 1, -1, 1]);
        assert.deepEqual(shortcuts.tools, []);
    }
});

test("letter shortcuts leave modified keys and workspace locks alone", () => {
    const shortcuts = installShortcuts();
    for (const option of [{ ctrlKey: true }, { metaKey: true }, { altKey: true }])
        for (const key of ["a", "d"])
            assert.equal(shortcuts.press(key, option).defaultPrevented, false);
    for (const tagName of ["INPUT", "TEXTAREA", "SELECT"])
        shortcuts.press("a", { target: { tagName } });
    assert.deepEqual(shortcuts.steps, []);

    for (const state of [{ exportOpen: true }, { propagating: true }]) {
        const locked = installShortcuts(state);
        locked.press("a");
        locked.press("d");
        assert.deepEqual(locked.steps, []);
    }
});

test("A, E and T do not switch tools", () => {
    const shortcuts = installShortcuts();
    for (const key of ["a", "e", "t"]) shortcuts.press(key);
    assert.deepEqual(shortcuts.tools, []);
});

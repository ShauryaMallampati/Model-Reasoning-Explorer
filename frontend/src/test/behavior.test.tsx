import React from "react";
import { MemoryRouter } from "react-router-dom";
import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, expect, test, vi } from "vitest";
import { getRun, getWsUrl, startRun } from "../api/client";
import AttributionText from "../components/run/AttributionText";
import CompareViewer from "../pages/CompareViewer";
import DatasetExplorer from "../pages/DatasetExplorer";

vi.mock("../components/common/Chart", () => ({ default: () => <div>Chart data</div> }));
afterEach(() => vi.unstubAllGlobals());

test("HTTP errors are rejected rather than interpreted as run data", async () => {
  vi.stubGlobal("fetch", vi.fn().mockResolvedValue({
    ok: false, status: 503, json: async () => ({ detail: "Model unavailable" })
  }));
  await expect(startRun({ task_type: "text_lm", model_id: "demo", input_text: "Hi" }))
    .rejects.toThrow("Model unavailable");
});

test("malformed server responses produce a visible error", async () => {
  vi.stubGlobal("fetch", vi.fn().mockResolvedValue({
    ok: false, status: 502, json: async () => { throw new Error("Invalid JSON"); }
  }));
  await expect(getRun("missing")).rejects.toThrow("unreadable response (502)");
});

test("WebSocket addresses are absolute and identifiers are encoded", () => {
  const url = new URL(getWsUrl("run_example"));
  expect(url.protocol).toBe("ws:");
  expect(url.pathname).toBe("/ws/runs/run_example");
});

test("attribution refuses mismatched tokens instead of inventing zero scores", () => {
  render(<AttributionText tokens={["one", "two"]} scores={[0.5]} />);
  expect(screen.getByRole("alert")).toHaveTextContent("do not match");
});

test("small attribution values retain their scale and token whitespace", () => {
  render(<AttributionText tokens={[" hello", "world"]} scores={[0.001, -0.002]} />);
  const spans = screen.getAllByTitle(/Attribution:/);
  expect(spans[0]).toHaveTextContent("hello");
  expect(spans[0].style.background).toBe("rgba(255, 124, 0, 0.5)");
  expect(spans[1].style.background).toBe("rgb(0, 148, 255)");
});

test("comparison errors are displayed and the action can be retried", async () => {
  vi.stubGlobal("fetch", vi.fn().mockResolvedValue({
    ok: false, status: 404, json: async () => ({ detail: "Run not found" })
  }));
  render(<CompareViewer />);
  fireEvent.change(screen.getByLabelText("Run A"), { target: { value: "first" } });
  fireEvent.change(screen.getByLabelText("Run B"), { target: { value: "second" } });
  fireEvent.click(screen.getByRole("button", { name: "Compare" }));
  expect(await screen.findByRole("alert")).toHaveTextContent("Run not found");
  expect(screen.getByRole("button", { name: "Compare" })).toBeEnabled();
});

test("dataset failures stop polling and restore the Run button", async () => {
  const fetchMock = vi.fn()
    .mockResolvedValueOnce({ ok: true, json: async () => ({ dataset_run_id: "dataset_test" }) })
    .mockResolvedValueOnce({ ok: true, json: async () => ({ status: "failed", error: "Invalid CSV", examples: [] }) });
  vi.stubGlobal("fetch", fetchMock);
  render(<MemoryRouter><DatasetExplorer /></MemoryRouter>);
  fireEvent.click(screen.getByRole("button", { name: "Run Dataset" }));
  expect(await screen.findByRole("alert")).toHaveTextContent("Invalid CSV");
  expect(screen.getByRole("button", { name: "Run Dataset" })).toBeEnabled();
  expect(fetchMock).toHaveBeenCalledTimes(2);
});

test("image datasets submit the image model rather than the text classifier", async () => {
  const fetchMock = vi.fn()
    .mockResolvedValueOnce({ ok: true, json: async () => ({ dataset_run_id: "dataset_image" }) })
    .mockResolvedValueOnce({ ok: true, json: async () => ({ status: "completed", examples: [] }) });
  vi.stubGlobal("fetch", fetchMock);
  render(<MemoryRouter><DatasetExplorer /></MemoryRouter>);
  fireEvent.change(screen.getByLabelText("Dataset Type"), { target: { value: "image" } });
  fireEvent.click(screen.getByRole("button", { name: "Run Dataset" }));
  await waitFor(() => expect(fetchMock).toHaveBeenCalledTimes(2));
  const form = fetchMock.mock.calls[0][1].body as FormData;
  expect(form.get("model_id")).toBe("resnet18");
  expect(form.get("path")).toBe("images");
});

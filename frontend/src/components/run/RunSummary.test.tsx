import { render, screen } from "@testing-library/react";
import RunSummary from "./RunSummary";

it("renders prediction", () => {
  render(<RunSummary outputs={{ prediction: "ok", top_k: [] }} metadata={{}} />);
  expect(screen.getByText("ok")).toBeInTheDocument();
});

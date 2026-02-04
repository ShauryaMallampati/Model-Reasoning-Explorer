import React from "react";

type NeuronExplorerProps = {
  activations?: { name: string; value: number }[];
};

const NeuronExplorer: React.FC<NeuronExplorerProps> = ({ activations = [] }) => {
  return (
    <section className="panel">
      <h3>Neuron / Feature Explorer</h3>
      <ul className="list">
        {activations.length === 0 && <li>No activations captured</li>}
        {activations.map((item) => (
          <li key={item.name}>
            <span>{item.name}</span>
            <span>{item.value.toFixed(4)}</span>
          </li>
        ))}
      </ul>
    </section>
  );
};

export default NeuronExplorer;

"""Malik et al. Figure 1 / Section 4: explicit six-qubit gate construction."""

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector

from src.config import CircuitConfig

READOUT_QUBITS = (3, 4, 5)


def build_malik_circuit(config: CircuitConfig | None = None) -> QuantumCircuit:
    """Return an unmeasured circuit with four x inputs and distinct theta weights.

    Default: two feature-map/ansatz blocks, three ansatz repetitions per block,
    72 weights and 48 CX gates. Changing repetitions or entanglement is an ablation.
    """
    config = config or CircuitConfig()
    inputs = ParameterVector("x", 4)
    weights = ParameterVector("theta", config.n_parameters)
    circuit = QuantumCircuit(6, name="malik_vqc")
    index = 0
    for _ in range(config.reuploads):
        for qubit in range(4):
            circuit.ry(inputs[qubit], qubit)
        if config.entanglement:
            for first, second in ((0, 1), (1, 2), (2, 3)):
                circuit.cx(first, second)
                circuit.rz(inputs[first] * inputs[second], second)
                circuit.cx(first, second)
        for _ in range(config.ansatz_reps):
            for qubit in range(6):
                circuit.ry(weights[index], qubit)
                circuit.rz(weights[index + 1], qubit)
                index += 2
            if config.entanglement:
                for qubit in range(6):
                    circuit.cx(qubit, (qubit + 1) % 6)
    return circuit


def parameter_groups(circuit: QuantumCircuit):
    """Qiskit orders ParameterVector elements by their numeric index."""
    inputs = [p for p in circuit.parameters if p.name.startswith("x[")]
    weights = [p for p in circuit.parameters if p.name.startswith("theta[")]
    return inputs, weights


def circuit_resources(circuit: QuantumCircuit) -> dict:
    operations = dict(circuit.count_ops())
    two_qubit = sum(instruction.operation.num_qubits == 2
                    for instruction in circuit.data if instruction.operation.name != "barrier")
    return {"qubits": circuit.num_qubits, "depth": circuit.depth(),
            "two_qubit_gates": int(two_qubit), "operations": operations,
            "parameters": circuit.num_parameters}

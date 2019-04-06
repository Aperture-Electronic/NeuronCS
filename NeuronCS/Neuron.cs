using NeuronCS.Functions;
using NeuronCS.Types;
using System;
using System.Collections.Generic;
using System.Text;

namespace NeuronCS.Neurons
{
    /// <summary>
    ///  A neuron, it is the most basic unit in the neural network.
    /// <para>You will get the output value by enter a set of values and a afferent function and a transfer function.</para>
    /// <para>Output = Transfer(Afferent(Input, Weight) + Bias)</para>
    /// </summary>
    public class Neuron
    {
        // Neuron inputs
        public double[] Inputs;
        public double[] Weights;

        // Neuron bias
        public double Bias;

        // Neuron functions
        private DoubleArrayFunction afferentFunction;
        private SingleValueFunction transferFunction;

        public DoubleArrayFunction AfferentFunction => afferentFunction;
        public SingleValueFunction TransferFunction => transferFunction;

        // Neuron value
        private double Value;

        /// <summary>
        /// Get the middle value of neuron : Middle = Afferent(Input, Weight) + Bias
        /// </summary>
        public double MiddleValue => Value;

        /// <summary>
        /// Get the output value of neuron
        /// </summary>
        public double OutputValue => transferFunction.Function(Value);

        // Constructor
        /// <summary>
        /// Create a neuron
        /// </summary>
        /// <param name="Inputs">Input values</param>
        /// <param name="Weights">Input weights</param>
        /// <param name="Bias">Bias</param>
        /// <param name="AfferentFunction">Afferent function: delegate f(Input, Weight, out Output)</param>
        /// <param name="TransferFunction">Transfer function: delegate g(Input, out Output)</param>
        public Neuron(double[] Inputs, double[] Weights, double Bias, DoubleArrayFunction AfferentFunction, SingleValueFunction TransferFunction)
        {
            afferentFunction = AfferentFunction;
            transferFunction = TransferFunction;

            Update(Inputs, Weights, Bias);
        }

        /// <summary>
        /// Update neuron immediately
        /// </summary>
        public void Update()
        {
            Value = afferentFunction.Function(Inputs, Weights) + Bias;
        }

        /// <summary>
        /// Update neuron immediately by new inputs
        /// </summary>
        /// <param name="Inputs">Input values</param>
        /// <param name="Weights">Input weights</param>
        /// <param name="Bias">Bias</param>
        public virtual void Update(double[] Inputs, double[] Weights, double Bias)
        {
            if ((Inputs != null) && (Weights != null))
            {
                // Check input
                int nI = Inputs.Length;
                int nW = Weights.Length;

                if (nI == 0) throw new ArgumentException("The size of the input array is invalid.", nameof(Inputs));
                if (nI != nW) throw new ArgumentException("The size of the input array and weight array are different.", $"{nameof(Inputs)} & {nameof(Weights)}");
            }
            
            // Write the values to private field
            if (Inputs != null) this.Inputs = Inputs;
            if (Weights != null) this.Weights = Weights;
            if (!double.IsNaN(Bias)) this.Bias = Bias;

            Update();
        }
    }

    /// <summary>
    /// A bias neuron, input always 1, and no self-bias.
    /// <para>Output = Bias</para>
    /// </summary>
    public class BiasNeuron : Neuron
    {
        /// <summary>
        /// Create a bias neuron
        /// </summary>
        /// <param name="Bias">Bias</param>
        public BiasNeuron() : base(new double[1] { 0 }, new double[1] { 0 }, 1, new InvalidDoubleArrayFunction(), new LinearFunction())
        {

        }

        // Deprecated
        [Obsolete("This is a empty method due to this method is not useful in the class", true)]
       new public void Update(double[] Inputs, double[] Weights, double Bias)
        {

        }
    }

    /// <summary>
    /// A input neuron with a normalization function and a input value
    /// <para>Output = Normalization(Input)</para>
    /// </summary>
    public class InputNeuron : Neuron
    {
        private double Input;

        /// <summary>
        /// Create a input neuron
        /// </summary>
        /// <param name="Input">Input</param>
        /// <param name="NormalizationFunction">Normalization function</param>
        public InputNeuron(double Input, SingleValueFunction NormalizationFunction) : base(new double[1] { Input }, new double[1] { 1 }, 0, new SumDoubleArrayFunction(), NormalizationFunction)
        {
            Update(Input);
        }

        // Deprecated
        [Obsolete("This is a empty method due to this method is not useful in the class", true)]
        new public void Update(double[] Inputs, double[] Weights, double Bias)
        {

        }

        /// <summary>
        /// Update neuron immediately by new input
        /// </summary>
        /// <param name="Input">Input</param>
        public void Update(double Input)
        {
            this.Input = Input;

            base.Update(new double[1] { Input }, new double[1] { 1 }, 0);
        }
    }
}

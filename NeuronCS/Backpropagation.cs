using NeuronCS.Dataset;
using NeuronCS.Types;
using NeuronCS.Neurons;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuronCS.Algorithm
{
    namespace Backpropagation
    {
        /// <summary>
        /// A unified function set of backpropagation network
        /// </summary>
        public struct BackpropagationNetworkFunctionsSet
        {
            /// <summary>
            /// A function to normalize the input data into a good range
            /// </summary>
            public SingleValueFunction InputNormalizationFunction;

            /// <summary>
            /// A afferent function to decide how to integrated all input values and weights in a neuron
            /// </summary>
            public DoubleArrayFunction AfferentFunction;

            /// <summary>
            /// A transfer function to decide how to transfer the middle value to neuron output
            /// </summary>
            public SingleValueFunction TransferFunction;
        };

        /// <summary>
        /// A unified full-connect backpropagation network, can be train by backpropagation method
        /// </summary>
        public class BackpropagationNetwork
        {
            /// <summary>
            /// [Readonly] Input neurons of network
            /// </summary>
            public readonly InputNeuron[] InputNeurons;

            /// <summary>
            /// [Readonly] Main neurons of network, list for the layers
            /// </summary>
            public readonly List<Neuron[]> Neurons;

            /// <summary>
            /// [Readonly] Bias neurons of network
            /// </summary>
            public readonly List<BiasNeuron> BiasNeurons;

            /// <summary>
            /// Return number of input neurons
            /// </summary>
            public int Inputs => InputNeurons.Length;

            /// <summary>
            /// Return number of layers
            /// </summary>
            public int Layers => Neurons.Count;

            /// <summary>
            /// Return how many neurons in the layer
            /// </summary>
            /// <param name="layer">Layer index</param>
            /// <returns></returns>
            public int GetNeuronsCount(int layer) => (Neurons[layer].Length);


            /// <summary>
            /// Initialize a new backpropagation(BP) network
            /// </summary>
            /// <param name="FunctionSet">Unified neuron functions</param>
            /// <param name="weightsRandomizer">Weights randomizer function (Current neuron, Neurons count, Previous layer neurons count)</param>
            /// <param name="Inputs">Input neurons count</param>
            /// <param name="Layers">Count of neurons in each layer</param>
            public BackpropagationNetwork(BackpropagationNetworkFunctionsSet FunctionSet, int Inputs, double[] WeightsRange, params int[] Layers)
            {
                InputNeurons = new InputNeuron[Inputs];
                for (int i = 0; i < Inputs; i++)
                {
                    InputNeurons[i] = new InputNeuron(0, FunctionSet.InputNormalizationFunction);
                }

                Neurons = new List<Neuron[]>();
                BiasNeurons = new List<BiasNeuron>();

                int layersCount = Layers.Length;
                for (int layer = 0; layer < layersCount; layer++)
                {
                    int prevLayerNeuronCount = (layer == 0) ? (Inputs) : (Layers[layer - 1]);
                    int neuronCount = Layers[layer];
                    IEnumerable<Neuron> prevLayer = (layer == 0) ? (InputNeurons) : (Neurons[layer - 1]);

                    BiasNeurons.Add(new BiasNeuron());

                    Neurons.Add(new Neuron[neuronCount]);

                    // Weight ramdomer
                    Random weightRamdom = new Random();

                    for (int j = 0; j < neuronCount; j++)
                    {
                        // Generate a new weights set 
                        double[] weights = new double[prevLayerNeuronCount];
                        for (int w = 0; w < prevLayerNeuronCount; w++)
                            weights[w] = WeightsRange[0] + weightRamdom.NextDouble() * (WeightsRange[1] - WeightsRange[0]);

                        double bias = WeightsRange[0] + weightRamdom.NextDouble() * (WeightsRange[1] - WeightsRange[0]);

                        double beta = 0.7 * Math.Pow(neuronCount + 1.0, 1.0 / Inputs);
                        double euclid = Math.Sqrt(weights.Select((double x) => (x * x)).Sum() + (bias * bias));

                        for (int w = 0; w < prevLayerNeuronCount; w++)
                            weights[w] = beta * weights[w] / euclid;

                        bias = beta * bias / euclid;

                        Neurons[layer][j] = new Neuron(prevLayer.Select((Neuron n) => (n.OutputValue)).ToArray(), weights, BiasNeurons[layer].OutputValue * bias, FunctionSet.AfferentFunction, FunctionSet.TransferFunction);
                    }
                }
            }

            /// <summary>
            /// Get network output (finally layer output)
            /// </summary>
            public double[] OutputValues => Neurons.Last().Select((Neuron n) => (n.OutputValue)).ToArray();

            /// <summary>
            /// Update full network use a set of new input values
            /// </summary>
            /// <param name="Input">New input values</param>
            public void UpdatePositive(double[] Input)
            {
                // Check parameters
                if (Input.Length != InputNeurons.Length) throw new ArgumentException("Invalid input array, its size is different from the number of input neurons.", nameof(Input));


                // Update input neurons
                for (int i = 0; i < InputNeurons.Length; i++)
                {
                    InputNeurons[i].Update(Input[i]);
                }

                // Sequential update neurons
                SequentialUpdate(0);
            }

            /// <summary>
            /// Sequential update the network from select layer
            /// </summary>
            /// <param name="StartLayer">Start layer</param>
            public void SequentialUpdate(int StartLayer)
            {
                for (int layer = StartLayer; layer < Neurons.Count; layer++)
                {
                    int neuronCount = Neurons[layer].Length;
                    IEnumerable<Neuron> prevLayer = (layer == 0) ? (InputNeurons) : (Neurons[layer - 1]);

                    for (int i = 0; i < neuronCount; i++)
                    {
                        Neuron neuron = Neurons[layer][i];

                        // Only update layer input
                        neuron.Update(prevLayer.Select((Neuron n) => (n.OutputValue)).ToArray(), null, double.NaN);
                    }
                }
            }

            /// <summary>
            /// Update the weights of a neuron
            /// </summary>
            /// <param name="Weights">Weights array</param>
            /// <param name="Layer">Layer index</param>
            /// <param name="Neuron">Neuron index</param>
            public void UpdateWeights(double[] Weights, int Layer, int Neuron)
            {
                // Check parameters
                if ((Layer < 0) || (Layer >= Neurons.Count)) throw new ArgumentException("Layer index out of range.", nameof(Layer));
                if ((Neuron < 0) || (Neuron > Neurons[Layer].Length)) throw new ArgumentException("Neuron index out of range.", nameof(Neuron));

                IEnumerable<Neuron> lastLayer = (Layer == 0) ? (InputNeurons) : (Neurons[Layer - 1]);

                if ((Weights.Length != lastLayer.ToArray().Length)) throw new ArgumentException("Size of weights array is different from previous layer's neurons count.", nameof(Weights));

                // Update weights
                Neuron update = Neurons[Layer][Neuron];
                update.Update(null, Weights, double.NaN);
            }

            /// <summary>
            /// Update the bias of layer
            /// </summary>
            /// <param name="Bias">New bias</param>
            /// <param name="Layer">Layer index</param>
            public void UpdateBias(double Bias, int Layer, int Neuron)
            {
                Neurons[Layer][Neuron].Update(null, null, Bias);
            }
        }

        /// <summary>
        /// A backpropagation network trainer use ada-gradient algorithm
        /// </summary>
        public class AdaTrainer
        {
            private BackpropagationNetwork TargetNetwork;
            private IOMetaDataSet<double[]> TrainingDataset;

            private AdaGradientDecent adaGradientDecent;

            /// <summary>
            /// Create a new ada-gradient network trainer
            /// </summary>
            /// <param name="TargetNetwork">Network to train</param>
            /// <param name="TrainingDataset">Data sets used for training</param>
            /// <param name="LearningRate">Initial learning rate</param>
            public AdaTrainer(BackpropagationNetwork TargetNetwork, IOMetaDataSet<double[]> TrainingDataset, double LearningRate)
            {
                this.TargetNetwork = TargetNetwork;
                this.TrainingDataset = TrainingDataset;

                adaGradientDecent = new AdaGradientDecent(LearningRate);
            }

            /// <summary>
            /// Get the mean square error by samples and corrects
            /// </summary>
            /// <param name="Sample">Samples</param>
            /// <param name="Correct">Correct values</param>
            /// <returns>Mean square error</returns>
            private double[] GetMeanSquareError(double[] Sample, double[] Correct)
            {
                return Enumerable.Zip(Sample, Correct, (double s, double c) => (c - s)).ToArray();
            }

            /// <summary>
            /// Get the current error of training dataset
            /// </summary>
            /// <returns>Errors of the set</returns>
            public double[] GetCurrentError()
            {
                double[] Errors = new double[TrainingDataset.Size];
                int i = 0;

                foreach(IOMetaDataSetItem<double[]> item in TrainingDataset)
                {
                    TargetNetwork.UpdatePositive(item.DataIn);

                    Errors[i] = 0.5 * GetMeanSquareError(TargetNetwork.OutputValues, item.DataOut).Select((double v) => (v * v)).Sum();
                    i++;
                }

                return Errors;
            }

            private void InitializeNetworkArray(ref List<List<double[]>> Array)
            {
                for (int layer = 0; layer < TargetNetwork.Layers; layer++)
                {
                    List<double[]> LayerArray = new List<double[]>();

                    // Get number of synapse pre neurons in this layer
                    int SynapseCount = (layer == 0) ? (TargetNetwork.Inputs) : (TargetNetwork.Neurons[layer - 1].Length);

                    // Get the number of neurons in this layer
                    int NeuronsCount = TargetNetwork.Neurons[layer].Length;

                    // Traversing all the neurons in this layer
                    for (int neuron = 0; neuron < NeuronsCount; neuron++)
                    {
                        double[] NeuronArray = new double[SynapseCount];
                        System.Array.Fill(NeuronArray, 0);
                        LayerArray.Add(NeuronArray);
                    }

                    Array.Add(LayerArray);
                }
            }

            private void InitializeBiasArray(ref List<double[]> Array)
            {
                for (int layer = 0; layer < TargetNetwork.Layers; layer++)
                {
                    // Get the number of neurons in this layer
                    int NeuronsCount = TargetNetwork.Neurons[layer].Length;

                    // Traversing all the neurons in this layer
                    double[] LayerArray = new double[NeuronsCount];
                    System.Array.Fill(LayerArray, 0);

                    Array.Add(LayerArray);
                }
            }

            /// <summary>
            /// Training the network
            /// </summary>
            /// <param name="Iteration">Number of iteration</param>
            /// <param name="MinimunError">Minimun error of training</param>
            /// <param name="AutoExit">Auto exit when error less then minimun error</param>
            /// <returns>A enumerable for errors</returns>
            public IEnumerable<double> Train(int Iteration, double MinimunError, bool AutoExit)
            {
                // A global gradient summary for ada-grad
                // double[] -- synapse of one neuron
                // List<double[]> -- neurons of one layer
                // List<List<double[]>> -- layers of the network
                List<List<double[]>> gradientSummary = new List<List<double[]>>();
                InitializeNetworkArray(ref gradientSummary);

                // A global gradient summary for ada-grad
                // double[] -- synapse of one layer of bias neuron
                // List<double[]> -- layers of the network
                List<double[]> biasGradientSummary = new List<double[]>();
                InitializeBiasArray(ref biasGradientSummary);

                for (int iteration = 0; iteration < Iteration; iteration++)
                {
                    foreach (IOMetaDataSetItem<double[]> item in TrainingDataset)
                    {
                        // A new weight array
                        List<List<double[]>> newWeights = new List<List<double[]>>();
                        InitializeNetworkArray(ref newWeights);

                        // A new bias array
                        List<double[]> newBias = new List<double[]>();
                        InitializeBiasArray(ref newBias);

                        // Update the network with sample
                        TargetNetwork.UpdatePositive(item.DataIn);

                        // Traversing all layers
                        double[] nexterrors = new double[0];
                        for (int layer = TargetNetwork.Layers - 1; layer >= 0; layer--)
                        {
                            // Get number of synapse pre neurons in this layer
                            int SynapseCount = (layer == 0) ? (TargetNetwork.Inputs) : (TargetNetwork.Neurons[layer - 1].Length);

                            // Get the number of neurons in this layer
                            int NeuronsCount = TargetNetwork.Neurons[layer].Length;

                            // Get the error from the back layer (backpropagation)
                            // When start (in the final layer), the error is network error
                            double[] errors = new double[NeuronsCount];

                            if (layer == TargetNetwork.Layers - 1)
                                errors = GetMeanSquareError(TargetNetwork.OutputValues, item.DataOut); // Start, final layer
                            else
                                errors = nexterrors;

                            // Create a new error array for next layer
                            nexterrors = new double[SynapseCount];
                            Array.Fill(nexterrors, 0);
                            // Create a variable to storage bias errors
                            // Traversing all the neurons in this layer
                            for (int neuron = 0; neuron < NeuronsCount; neuron++)
                            {
                                Neuron n = TargetNetwork.Neurons[layer][neuron];

                                // Traversing all synapse in this neuron and calculate the gradients
                                double[] weights = n.Weights;
                                double[] gradients = new double[SynapseCount];
                                double[] summary = gradientSummary[layer][neuron];
                                for (int synapse = 0; synapse < SynapseCount; synapse++)
                                {
                                    double prev = (layer == 0) ? (TargetNetwork.InputNeurons[synapse].OutputValue) : (TargetNetwork.Neurons[layer - 1][synapse].OutputValue);

                                    // Calculate gradient => g = - E * d(f(Is))/d(Is) * Os
                                    double gradient = -errors[neuron] * n.TransferFunction.Derivatives(n.MiddleValue) * prev;

                                    // Save the gradient
                                    gradients[synapse] = gradient;

                                    // Statistic next error => next error = Σ(Ws * e)
                                    nexterrors[synapse] += TargetNetwork.Neurons[layer][neuron].Weights[synapse] * errors[neuron];
                                }

                                // Gradient decent
                                object unused = new object();
                                adaGradientDecent.Update(ref weights, gradients, ref summary, null, ref unused);

                                // Update the summary
                                gradientSummary[layer][neuron] = summary;

                                // Save the new weights
                                newWeights[layer][neuron] = weights;
                            }

                            // Now calculate the bias, but only apart from the last layer have bias
                            // The bias[a] is actually connect to layer[b]
                            // We had get the errors from neurons calculation
                            double[] bias = TargetNetwork.Neurons[layer].Select((Neuron p) => (p.Bias)).ToArray();
                            double[] biasGradients = new double[NeuronsCount];
                            double[] biasSummary = biasGradientSummary[layer];
                            for (int synapse = 0; synapse < NeuronsCount; synapse++)
                            {
                                // Calculate gradient => g = - E * Os
                                double gradient = -errors[synapse] * TargetNetwork.BiasNeurons[layer].OutputValue;

                                // Save the gradient
                                biasGradients[synapse] = gradient;
                            }

                            // Gradient decent
                            object biasUnused = new object();
                            adaGradientDecent.Update(ref bias, biasGradients, ref biasSummary, null, ref biasUnused);

                            // Update the summary
                            biasGradientSummary[layer] = biasSummary;

                            // Save the new bias
                            newBias[layer] = bias;
                        }

                        // Update the weights by new weights
                        for (int layer = 0; layer < TargetNetwork.Layers; layer++)
                        {
                            // Get number of synapse pre neurons in this layer
                            int SynapseCount = (layer == 0) ? (TargetNetwork.Inputs) : (TargetNetwork.Neurons[layer - 1].Length);

                            // Get the number of neurons in this layer
                            int NeuronsCount = TargetNetwork.Neurons[layer].Length;

                            // Traversing all the neurons in this layer
                            for (int neuron = 0; neuron < NeuronsCount; neuron++)
                            {
                                TargetNetwork.Neurons[layer][neuron].Weights = newWeights[layer][neuron];
                                TargetNetwork.Neurons[layer][neuron].Bias = newBias[layer][neuron];
                            }
                        }
                    }

                    double error = GetCurrentError().Average();
                    yield return error;
                    if (error <= MinimunError)
                        if (AutoExit == true)
                            break;
                }
            }
        }

        /// <summary>
        /// A backpropagation network trainer use ada-delta algorithm
        /// </summary>
        public class AdaDeltaTrainer
        {
            BackpropagationNetwork TargetNetwork;
            IOMetaDataSet<double[]> TrainingDataset;

            AdaDeltaDecent adaDeltaDecent;

            /// <summary>
            /// Create a new ada-gradient network trainer
            /// </summary>
            /// <param name="TargetNetwork">Network to train</param>
            /// <param name="TrainingDataset">Data sets used for training</param>
            /// <param name="MeanFactor">Mean factor</param>
            /// <param name="LearningRate">Initial learning rate, default = 0.1</param>
            public AdaDeltaTrainer(BackpropagationNetwork TargetNetwork, IOMetaDataSet<double[]> TrainingDataset, double MeanFactor, double LearningRate = 0.1)
            {
                this.TargetNetwork = TargetNetwork;
                this.TrainingDataset = TrainingDataset;

                adaDeltaDecent = new AdaDeltaDecent(LearningRate, MeanFactor);
            }

            /// <summary>
            /// Get the mean square error by samples and corrects
            /// </summary>
            /// <param name="Sample">Samples</param>
            /// <param name="Correct">Correct values</param>
            /// <returns>Mean square error</returns>
            private double[] GetMeanSquareError(double[] Sample, double[] Correct)
            {
                return Enumerable.Zip(Sample, Correct, (double s, double c) => (c - s)).ToArray();
            }

            /// <summary>
            /// Get the current error of training dataset
            /// </summary>
            /// <returns>Errors of the set</returns>
            public double[] GetCurrentError()
            {
                double[] Errors = new double[TrainingDataset.Size];
                int i = 0;

                foreach (IOMetaDataSetItem<double[]> item in TrainingDataset)
                {
                    TargetNetwork.UpdatePositive(item.DataIn);

                    Errors[i] = 0.5 * GetMeanSquareError(TargetNetwork.OutputValues, item.DataOut).Select((double v) => (v * v)).Sum();
                    i++;
                }

                return Errors;
            }

            private void InitializeNetworkArray(ref List<List<double[]>> Array)
            {
                for (int layer = 0; layer < TargetNetwork.Layers; layer++)
                {
                    List<double[]> LayerArray = new List<double[]>();

                    // Get number of synapse pre neurons in this layer
                    int SynapseCount = (layer == 0) ? (TargetNetwork.Inputs) : (TargetNetwork.Neurons[layer - 1].Length);

                    // Get the number of neurons in this layer
                    int NeuronsCount = TargetNetwork.Neurons[layer].Length;

                    // Traversing all the neurons in this layer
                    for (int neuron = 0; neuron < NeuronsCount; neuron++)
                    {
                        double[] NeuronArray = new double[SynapseCount];
                        System.Array.Fill(NeuronArray, 0);
                        LayerArray.Add(NeuronArray);
                    }

                    Array.Add(LayerArray);
                }
            }

            private void InitializeBiasArray(ref List<double[]> Array)
            {
                for (int layer = 0; layer < TargetNetwork.Layers; layer++)
                {
                    // Get the number of neurons in this layer
                    int NeuronsCount = TargetNetwork.Neurons[layer].Length;

                    // Traversing all the neurons in this layer
                    double[] LayerArray = new double[NeuronsCount];
                    System.Array.Fill(LayerArray, 0);

                    Array.Add(LayerArray);
                }
            }

            /// <summary>
            /// Training the network
            /// </summary>
            /// <param name="Iteration">Number of iteration</param>
            /// <param name="MinimunError">Minimun error of training</param>
            /// <param name="AutoExit">Auto exit when error less then minimun error</param>
            /// <returns>A enumerable for errors</returns>
            public IEnumerable<double> Train(int Iteration, double MinimunError, bool AutoExit)
            {
                // A global gradient summary for ada-grad
                // double[] -- synapse of one neuron
                // List<double[]> -- neurons of one layer
                // List<List<double[]>> -- layers of the network
                List<List<double[]>> gradientSummary = new List<List<double[]>>();
                List<List<double[]>> gradientPrevSummary = new List<List<double[]>>();
                InitializeNetworkArray(ref gradientSummary);
                InitializeNetworkArray(ref gradientPrevSummary);

                // A global gradient summary for ada-grad
                // double[] -- synapse of one layer of bias neuron
                // List<double[]> -- layers of the network
                List<double[]> biasGradientSummary = new List<double[]>();
                List<double[]> biasPrevGradientSummary = new List<double[]>();
                InitializeBiasArray(ref biasGradientSummary);
                InitializeBiasArray(ref biasPrevGradientSummary);

                for (int iteration = 0; iteration < Iteration; iteration++)
                {
                    foreach (IOMetaDataSetItem<double[]> item in TrainingDataset)
                    {
                        // A new weight array
                        List<List<double[]>> newWeights = new List<List<double[]>>();
                        InitializeNetworkArray(ref newWeights);

                        // A new bias array
                        List<double[]> newBias = new List<double[]>();
                        InitializeBiasArray(ref newBias);

                        // Update the network with sample
                        TargetNetwork.UpdatePositive(item.DataIn);

                        // Traversing all layers
                        double[] nexterrors = new double[0];
                        for (int layer = TargetNetwork.Layers - 1; layer >= 0; layer--)
                        {
                            // Get number of synapse pre neurons in this layer
                            int SynapseCount = (layer == 0) ? (TargetNetwork.Inputs) : (TargetNetwork.Neurons[layer - 1].Length);

                            // Get the number of neurons in this layer
                            int NeuronsCount = TargetNetwork.Neurons[layer].Length;

                            // Get the error from the back layer (backpropagation)
                            // When start (in the final layer), the error is network error
                            double[] errors = new double[NeuronsCount];

                            if (layer == TargetNetwork.Layers - 1)
                                errors = GetMeanSquareError(TargetNetwork.OutputValues, item.DataOut); // Start, final layer
                            else
                                errors = nexterrors;

                            // Create a new error array for next layer
                            nexterrors = new double[SynapseCount];
                            Array.Fill(nexterrors, 0);
                            // Create a variable to storage bias errors
                            // Traversing all the neurons in this layer
                            for (int neuron = 0; neuron < NeuronsCount; neuron++)
                            {
                                Neuron n = TargetNetwork.Neurons[layer][neuron];

                                // Traversing all synapse in this neuron and calculate the gradients
                                double[] weights = n.Weights;
                                double[] gradients = new double[SynapseCount];
                                double[] summary = gradientSummary[layer][neuron];
                                for (int synapse = 0; synapse < SynapseCount; synapse++)
                                {
                                    double prev = (layer == 0) ? (TargetNetwork.InputNeurons[synapse].OutputValue) : (TargetNetwork.Neurons[layer - 1][synapse].OutputValue);

                                    // Calculate gradient => g = - E * d(f(Is))/d(Is) * Os
                                    double gradient = -errors[neuron] * n.TransferFunction.Derivatives(n.MiddleValue) * prev;

                                    // Save the gradient
                                    gradients[synapse] = gradient;

                                    // Statistic next error => next error = Σ(Ws * e)
                                    nexterrors[synapse] += TargetNetwork.Neurons[layer][neuron].Weights[synapse] * errors[neuron];
                                }

                                // Gradient decent
                                object previousGradient = gradientPrevSummary[layer][neuron];
                                adaDeltaDecent.Update(ref weights, gradients, ref summary, null, ref previousGradient);

                                // Update the summary
                                gradientSummary[layer][neuron] = summary;
                                gradientPrevSummary[layer][neuron] = previousGradient as double[];

                                // Save the new weights
                                newWeights[layer][neuron] = weights;
                            }

                            // Now calculate the bias, but only apart from the last layer have bias
                            // The bias[a] is actually connect to layer[b]
                            // We had get the errors from neurons calculation
                            double[] bias = TargetNetwork.Neurons[layer].Select((Neuron p) => (p.Bias)).ToArray();
                            double[] biasGradients = new double[NeuronsCount];
                            double[] biasSummary = biasGradientSummary[layer];
                            for (int synapse = 0; synapse < NeuronsCount; synapse++)
                            {
                                // Calculate gradient => g = - E * Os
                                double gradient = -errors[synapse] * TargetNetwork.BiasNeurons[layer].OutputValue;

                                // Save the gradient
                                biasGradients[synapse] = gradient;
                            }

                            // Gradient decent
                            object biasPreviousGradient = biasPrevGradientSummary[layer];
                            adaDeltaDecent.Update(ref bias, biasGradients, ref biasSummary, null, ref biasPreviousGradient);

                            // Update the summary
                            biasGradientSummary[layer] = biasSummary;
                            biasPrevGradientSummary[layer] = biasPreviousGradient as double[];

                            // Save the new bias
                            newBias[layer] = bias;
                        }

                        // Update the weights by new weights
                        for (int layer = 0; layer < TargetNetwork.Layers; layer++)
                        {
                            // Get number of synapse pre neurons in this layer
                            int SynapseCount = (layer == 0) ? (TargetNetwork.Inputs) : (TargetNetwork.Neurons[layer - 1].Length);

                            // Get the number of neurons in this layer
                            int NeuronsCount = TargetNetwork.Neurons[layer].Length;

                            // Traversing all the neurons in this layer
                            for (int neuron = 0; neuron < NeuronsCount; neuron++)
                            {
                                TargetNetwork.Neurons[layer][neuron].Weights = newWeights[layer][neuron];
                                TargetNetwork.Neurons[layer][neuron].Bias = newBias[layer][neuron];
                            }
                        }
                    }

                    double error = GetCurrentError().Average();
                    yield return error;
                    if (error <= MinimunError)
                        if (AutoExit == true)
                            break;
                }
            }
        }

        /// <summary>
        /// A backpropagation network trainer use adam algorithm
        /// </summary>
        public class AdamTrainer
        {
            /// <summary>
            /// Target network
            /// </summary>
            public BackpropagationNetwork TargetNetwork;
            /// <summary>
            /// Training dataset
            /// </summary>
            public IOMetaDataSet<double[]> TrainingDataset;

            /// <summary>
            /// Adam-decent  algorithm
            /// </summary>
            public AdamDecent adamDecent;

            /// <summary>
            /// Create a new adam network trainer
            /// </summary>
            /// <param name="TargetNetwork">Network to train</param>
            /// <param name="TrainingDataset">Data sets used for training</param>
            /// <param name="LearningRate">Initial learning rate, default = 0.1</param>
            public AdamTrainer(BackpropagationNetwork TargetNetwork, IOMetaDataSet<double[]> TrainingDataset, double LearningRate = 0.1)
            {
                this.TargetNetwork = TargetNetwork;
                this.TrainingDataset = TrainingDataset;

                adamDecent = new AdamDecent(LearningRate);
            }

            /// <summary>
            /// Get the mean square error by samples and corrects
            /// </summary>
            /// <param name="Sample">Samples</param>
            /// <param name="Correct">Correct values</param>
            /// <returns>Mean square error</returns>
            public double[] GetMeanSquareError(double[] Sample, double[] Correct)
            {
                return Enumerable.Zip(Sample, Correct, (double s, double c) => (c - s)).ToArray();
            }

            /// <summary>
            /// Get the current error of training dataset
            /// </summary>
            /// <returns>Errors of the set</returns>
            public double[] GetCurrentError()
            {
                double[] Errors = new double[TrainingDataset.Size];
                int i = 0;

                foreach (IOMetaDataSetItem<double[]> item in TrainingDataset)
                {
                    TargetNetwork.UpdatePositive(item.DataIn);

                    Errors[i] = 0.5 * GetMeanSquareError(TargetNetwork.OutputValues, item.DataOut).Select((double v) => (v * v)).Sum();
                    i++;
                }

                return Errors;
            }

            /// <summary>
            /// Initialize a empty array for network
            /// </summary>
            /// <param name="Array">Reference of the array</param>
            public void InitializeNetworkArray(ref List<List<double[]>> Array)
            {
                for (int layer = 0; layer < TargetNetwork.Layers; layer++)
                {
                    List<double[]> LayerArray = new List<double[]>();

                    // Get number of synapse pre neurons in this layer
                    int SynapseCount = (layer == 0) ? (TargetNetwork.Inputs) : (TargetNetwork.Neurons[layer - 1].Length);

                    // Get the number of neurons in this layer
                    int NeuronsCount = TargetNetwork.Neurons[layer].Length;

                    // Traversing all the neurons in this layer
                    for (int neuron = 0; neuron < NeuronsCount; neuron++)
                    {
                        double[] NeuronArray = new double[SynapseCount];
                        System.Array.Fill(NeuronArray, 0);
                        LayerArray.Add(NeuronArray);
                    }

                    Array.Add(LayerArray);
                }
            }

            /// <summary>
            /// Initialize a empty array for bias
            /// </summary>
            /// <param name="Array">Reference of the array</param>
            public void InitializeBiasArray(ref List<double[]> Array)
            {
                for (int layer = 0; layer < TargetNetwork.Layers; layer++)
                {
                    // Get the number of neurons in this layer
                    int NeuronsCount = TargetNetwork.Neurons[layer].Length;

                    // Traversing all the neurons in this layer
                    double[] LayerArray = new double[NeuronsCount];
                    System.Array.Fill(LayerArray, 0);

                    Array.Add(LayerArray);
                }
            }

            /// <summary>
            /// Training the network
            /// </summary>
            /// <param name="Iteration">Number of iteration</param>
            /// <param name="MinimunError">Minimun error of training</param>
            /// <param name="AutoExit">Auto exit when error less then minimun error</param>
            /// <returns>A enumerable for errors</returns>
            public IEnumerable<double> Train(int Iteration, double MinimunError, bool AutoExit)
            {
                // A global gradient summary for adam-grad
                // double[] -- synapse of one neuron
                // List<double[]> -- neurons of one layer
                // List<List<double[]>> -- layers of the network
                List<List<double[]>> MSummary = new List<List<double[]>>();
                List<List<double[]>> NSummary = new List<List<double[]>>();
                InitializeNetworkArray(ref MSummary);
                InitializeNetworkArray(ref NSummary);

                // A global gradient summary for adam-grad
                // double[] -- synapse of one layer of bias neuron
                // List<double[]> -- layers of the network
                List<double[]> biasMSummary = new List<double[]>();
                List<double[]> biasNSummary = new List<double[]>();
                InitializeBiasArray(ref biasMSummary);
                InitializeBiasArray(ref biasNSummary);

                adamDecent.ClearCorrectAccumulate();

                for (int iteration = 0; iteration < Iteration; iteration++)
                {
                    foreach (IOMetaDataSetItem<double[]> item in TrainingDataset)
                    {
                        // A new weight array
                        List<List<double[]>> newWeights = new List<List<double[]>>();
                        InitializeNetworkArray(ref newWeights);

                        // A new bias array
                        List<double[]> newBias = new List<double[]>();
                        InitializeBiasArray(ref newBias);

                        // Update the network with sample
                        TargetNetwork.UpdatePositive(item.DataIn);

                        // Traversing all layers
                        double[] nexterrors = new double[0];
                        for (int layer = TargetNetwork.Layers - 1; layer >= 0; layer--)
                        {
                            // Get number of synapse pre neurons in this layer
                            int SynapseCount = (layer == 0) ? (TargetNetwork.Inputs) : (TargetNetwork.Neurons[layer - 1].Length);

                            // Get the number of neurons in this layer
                            int NeuronsCount = TargetNetwork.Neurons[layer].Length;

                            // Get the error from the back layer (backpropagation)
                            // When start (in the final layer), the error is network error
                            double[] errors = new double[NeuronsCount];

                            if (layer == TargetNetwork.Layers - 1)
                                errors = GetMeanSquareError(TargetNetwork.OutputValues, item.DataOut); // Start, final layer
                            else
                                errors = nexterrors;

                            // Create a new error array for next layer
                            nexterrors = new double[SynapseCount];
                            Array.Fill(nexterrors, 0);
                            // Create a variable to storage bias errors
                            // Traversing all the neurons in this layer
                            for (int neuron = 0; neuron < NeuronsCount; neuron++)
                            {
                                Neuron n = TargetNetwork.Neurons[layer][neuron];

                                // Traversing all synapse in this neuron and calculate the gradients
                                double[] weights = n.Weights;
                                double[] gradients = new double[SynapseCount];
                                double[] msummary = MSummary[layer][neuron];
                                double[] nsummary = NSummary[layer][neuron];
                                for (int synapse = 0; synapse < SynapseCount; synapse++)
                                {
                                    double prev = (layer == 0) ? (TargetNetwork.InputNeurons[synapse].OutputValue) : (TargetNetwork.Neurons[layer - 1][synapse].OutputValue);

                                    // Calculate gradient => g = - E * d(f(Is))/d(Is) * Os
                                    double gradient = -errors[neuron] * n.TransferFunction.Derivatives(n.MiddleValue) * prev;

                                    // Check the gradient to avoid NaN values
                                    if (double.IsNaN(gradient)) throw new Exception("Gradient is NaN");

                                    // Save the gradient
                                    gradients[synapse] = gradient;

                                    // Statistic next error => next error = Σ(Ws * e)
                                    nexterrors[synapse] += TargetNetwork.Neurons[layer][neuron].Weights[synapse] * errors[neuron];
                                }

                                // Gradient decent
                                double[] biasunused = new double[0];
                                object biasparameter = new List<double[]>() { msummary, nsummary };
                                adamDecent.Update(ref weights, gradients, ref biasunused, null, ref biasparameter);

                                // Update the summary
                                MSummary[layer][neuron] = (biasparameter as List<double[]>)[0];
                                NSummary[layer][neuron] = (biasparameter as List<double[]>)[1];

                                // Save the new weights
                                newWeights[layer][neuron] = weights;
                            }

                            // Now calculate the bias, but only apart from the last layer have bias
                            // The bias[a] is actually connect to layer[b]
                            // We had get the errors from neurons calculation
                            double[] bias = TargetNetwork.Neurons[layer].Select((Neuron p) => (p.Bias)).ToArray();
                            double[] biasGradients = new double[NeuronsCount];
                            double[] biasmSummary = biasMSummary[layer];
                            double[] biasnSummary = biasNSummary[layer];
                            for (int synapse = 0; synapse < NeuronsCount; synapse++)
                            {
                                // Calculate gradient => g = - E * Os
                                double gradient = -errors[synapse] * TargetNetwork.BiasNeurons[layer].OutputValue;

                                // Save the gradient
                                biasGradients[synapse] = gradient;
                            }

                            // Gradient decent
                            double[] unused = new double[0];
                            object parameter = new List<double[]>() { biasmSummary, biasnSummary };
                            adamDecent.Update(ref bias, biasGradients, ref unused, null, ref parameter);

                            // Update the summary
                            biasMSummary[layer] = (parameter as List<double[]>)[0];
                            biasNSummary[layer] = (parameter as List<double[]>)[1];

                            // Save the new bias
                            newBias[layer] = bias;
                        }

                        // Update the weights by new weights
                        for (int layer = 0; layer < TargetNetwork.Layers; layer++)
                        {
                            // Get number of synapse pre neurons in this layer
                            int SynapseCount = (layer == 0) ? (TargetNetwork.Inputs) : (TargetNetwork.Neurons[layer - 1].Length);

                            // Get the number of neurons in this layer
                            int NeuronsCount = TargetNetwork.Neurons[layer].Length;

                            // Traversing all the neurons in this layer
                            for (int neuron = 0; neuron < NeuronsCount; neuron++)
                            {
                                TargetNetwork.Neurons[layer][neuron].Weights = newWeights[layer][neuron];
                                TargetNetwork.Neurons[layer][neuron].Bias = newBias[layer][neuron];
                            }
                        }

                        // Correct accumulate
                        adamDecent.CorrectAccumulate();
                    }

                    // Return the deviation of the new iteration
                    double error = GetCurrentError().Average();
                    yield return error;
                    if (error <= MinimunError)
                        if (AutoExit == true)
                            break;
                }
            }
        }
    }
}

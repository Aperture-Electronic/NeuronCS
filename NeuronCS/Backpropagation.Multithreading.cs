using NeuronCS.Algorithm.Backpropagation;
using NeuronCS.Dataset;
using NeuronCS.Neurons;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuronCS
{
    namespace Backpropagation
    {
        namespace Multitherading
        {
            public static class BackpropagationNetworkMultitherading
            {
                public static void UpdatePositiveMultitherading(this BackpropagationNetwork backpropagation, double[] Input)
                {
                    // Check parameters
                    if (Input.Length != backpropagation.InputNeurons.Length) throw new ArgumentException("Invalid input array, its size is different from the number of input neurons.", nameof(Input));


                    // Update input neurons
                    Parallel.For(0, backpropagation.InputNeurons.Length, delegate (int i)
                    {
                        backpropagation.InputNeurons[i].Update(Input[i]);
                    });

                    // Sequential update neurons
                    backpropagation.SequentialUpdateMultitherading(0);
                }

                /// <summary>
                /// Sequential update the network from select layer with multitherading
                /// </summary>
                /// <param name="StartLayer">Start layer</param>
                public static void SequentialUpdateMultitherading(this BackpropagationNetwork backpropagation, int StartLayer)
                {
                    for (int layer = StartLayer; layer < backpropagation.Neurons.Count; layer++)
                    {
                        int neuronCount = backpropagation.Neurons[layer].Length;
                        IEnumerable<Neuron> prevLayer = (layer == 0) ? (backpropagation.InputNeurons) : (backpropagation.Neurons[layer - 1]);

                        Parallel.For(0, neuronCount, delegate (int i)
                        {
                            Neuron neuron = backpropagation.Neurons[layer][i];

                            neuron.Update(prevLayer.Select((Neuron n) => (n.OutputValue)).ToArray(), null, double.NaN);
                        });
                    }
                }
            }

            public static class AdamTrainerMultitherading
            {
                /// <summary>
                /// Get the current error of training dataset with multitherading
                /// </summary>
                /// <returns>Errors of the set</returns>
                public static double[] GetCurrentErrorMultitherading(this AdamTrainer adam)
                {
                    double[] Errors = new double[adam.TrainingDataset.Size];
                    int i = 0;

                    foreach(IOMetaDataSetItem<double[]> item in adam.TrainingDataset)
                    {
                        adam.TargetNetwork.UpdatePositiveMultitherading(item.DataIn);

                        Errors[i] = 0.5 * adam.GetMeanSquareError(adam.TargetNetwork.OutputValues, item.DataOut).Select((double v) => (v * v)).Sum();
                        i++;
                    }

                    return Errors;
                }

                /// <summary>
                /// Training the network with multitherading
                /// </summary>
                /// <param name="Iteration">Number of iteration</param>
                /// <param name="MinimunError">Minimun error of training</param>
                /// <param name="AutoExit">Auto exit when error less then minimun error</param>
                /// <returns>A enumerable for errors</returns>
                public static IEnumerable<double> TrainMultitherading(this AdamTrainer adam, int Iteration, double MinimunError, bool AutoExit)
                {
                    // A global gradient summary for adam-grad
                    // double[] -- synapse of one neuron
                    // List<double[]> -- neurons of one layer
                    // List<List<double[]>> -- layers of the network
                    List<List<double[]>> MSummary = new List<List<double[]>>();
                    List<List<double[]>> NSummary = new List<List<double[]>>();
                    adam.InitializeNetworkArray(ref MSummary);
                    adam.InitializeNetworkArray(ref NSummary);

                    // A global gradient summary for adam-grad
                    // double[] -- synapse of one layer of bias neuron
                    // List<double[]> -- layers of the network
                    List<double[]> biasMSummary = new List<double[]>();
                    List<double[]> biasNSummary = new List<double[]>();
                    adam.InitializeBiasArray(ref biasMSummary);
                    adam.InitializeBiasArray(ref biasNSummary);

                    adam.adamDecent.ClearCorrectAccumulate();

                    for (int iteration = 0; iteration < Iteration; iteration++)
                    {
                        foreach (IOMetaDataSetItem<double[]> item in adam.TrainingDataset)
                        {
                            // A new weight array
                            List<List<double[]>> newWeights = new List<List<double[]>>();
                            adam.InitializeNetworkArray(ref newWeights);

                            // A new bias array
                            List<double[]> newBias = new List<double[]>();
                            adam.InitializeBiasArray(ref newBias);

                            // Update the network with sample
                            adam.TargetNetwork.UpdatePositiveMultitherading(item.DataIn);

                            // Traversing all layers
                            double[] nexterrors = new double[0];

                            for (int layer = adam.TargetNetwork.Layers - 1; layer >= 0; layer--)
                            {
                                // Get number of synapse pre neurons in this layer
                                int SynapseCount = (layer == 0) ? (adam.TargetNetwork.Inputs) : (adam.TargetNetwork.Neurons[layer - 1].Length);

                                // Get the number of neurons in this layer
                                int NeuronsCount = adam.TargetNetwork.Neurons[layer].Length;

                                // Get the error from the back layer (backpropagation)
                                // When start (in the final layer), the error is network error
                                double[] errors = new double[NeuronsCount];

                                if (layer == adam.TargetNetwork.Layers - 1)
                                    errors = adam.GetMeanSquareError(adam.TargetNetwork.OutputValues, item.DataOut); // Start, final layer
                                else
                                    errors = nexterrors;

                                // Create a new error array for next layer
                                nexterrors = new double[SynapseCount];
                                Array.Fill(nexterrors, 0);
                                // Create a variable to storage bias errors
                                // Traversing all the neurons in this layer
                                Parallel.For(0, NeuronsCount, delegate (int neuron)
                                { 
                                    Neuron n = adam.TargetNetwork.Neurons[layer][neuron];

                                    // Traversing all synapse in this neuron and calculate the gradients
                                    double[] weights = n.Weights;
                                    double[] gradients = new double[SynapseCount];
                                    double[] msummary = MSummary[layer][neuron];
                                    double[] nsummary = NSummary[layer][neuron];

                                    for (int synapse = 0; synapse < SynapseCount; synapse++)
                                    {
                                        double prev = (layer == 0) ? (adam.TargetNetwork.InputNeurons[synapse].OutputValue) : (adam.TargetNetwork.Neurons[layer - 1][synapse].OutputValue);

                                        // Calculate gradient => g = - E * d(f(Is))/d(Is) * Os
                                        double gradient = -errors[neuron] * n.TransferFunction.Derivatives(n.MiddleValue) * prev;

                                        // Check the gradient to avoid NaN values
                                        if (double.IsNaN(gradient)) throw new Exception("Gradient is NaN");

                                        // Save the gradient
                                        gradients[synapse] = gradient;

                                        // Statistic next error => next error = Σ(Ws * e)
                                        nexterrors[synapse] += adam.TargetNetwork.Neurons[layer][neuron].Weights[synapse] * errors[neuron];
                                    }

                                    // Gradient decent
                                    double[] biasunused = new double[0];
                                    object biasparameter = new List<double[]>() { msummary, nsummary };
                                    adam.adamDecent.Update(ref weights, gradients, ref biasunused, null, ref biasparameter);

                                    // Update the summary
                                    MSummary[layer][neuron] = (biasparameter as List<double[]>)[0];
                                    NSummary[layer][neuron] = (biasparameter as List<double[]>)[1];

                                    // Save the new weights
                                    newWeights[layer][neuron] = weights;
                                });

                                // Now calculate the bias, but only apart from the last layer have bias
                                // The bias[a] is actually connect to layer[b]
                                // We had get the errors from neurons calculation
                                double[] bias = adam.TargetNetwork.Neurons[layer].Select((Neuron p) => (p.Bias)).ToArray();
                                double[] biasGradients = new double[NeuronsCount];
                                double[] biasmSummary = biasMSummary[layer];
                                double[] biasnSummary = biasNSummary[layer];
                                for (int synapse = 0; synapse < NeuronsCount; synapse++)
                                {
                                    // Calculate gradient => g = - E * Os
                                    double gradient = -errors[synapse] * adam.TargetNetwork.BiasNeurons[layer].OutputValue;

                                    // Save the gradient
                                    biasGradients[synapse] = gradient;
                                }

                                // Gradient decent
                                double[] unused = new double[0];
                                object parameter = new List<double[]>() { biasmSummary, biasnSummary };
                                adam.adamDecent.Update(ref bias, biasGradients, ref unused, null, ref parameter);

                                // Update the summary
                                biasMSummary[layer] = (parameter as List<double[]>)[0];
                                biasNSummary[layer] = (parameter as List<double[]>)[1];

                                // Save the new bias
                                newBias[layer] = bias;
                            }

                            // Update the weights by new weights
                            for (int layer = 0; layer < adam.TargetNetwork.Layers; layer++)
                            {
                                // Get number of synapse pre neurons in this layer
                                int SynapseCount = (layer == 0) ? (adam.TargetNetwork.Inputs) : (adam.TargetNetwork.Neurons[layer - 1].Length);

                                // Get the number of neurons in this layer
                                int NeuronsCount = adam.TargetNetwork.Neurons[layer].Length;

                                // Traversing all the neurons in this layer
                                for (int neuron = 0; neuron < NeuronsCount; neuron++)
                                {
                                    adam.TargetNetwork.Neurons[layer][neuron].Weights = newWeights[layer][neuron];
                                    adam.TargetNetwork.Neurons[layer][neuron].Bias = newBias[layer][neuron];
                                }
                            }

                            // Correct accumulate
                            adam.adamDecent.CorrectAccumulate();
                        }

                        // Return the deviation of the new iteration
                        double error = adam.GetCurrentErrorMultitherading().Average();
                        yield return error;
                        if (error <= MinimunError)
                            if (AutoExit == true)
                                break;
                    }
                }
            }
        }
    }
}

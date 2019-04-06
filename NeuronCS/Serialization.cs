using NeuronCS.Algorithm.Backpropagation;
using System;
using System.Collections.Generic;
using System.IO;
using System.Text;

namespace NeuronCS.Serialization
{
    class BackpropagationNetworkSerializer
    {
        private BackpropagationNetwork targetNetwork;

        public BackpropagationNetworkSerializer(BackpropagationNetwork TargetNetwork)
        {
            targetNetwork = TargetNetwork;
        }

        public void Serialize(Stream stream)
        {
            BinaryWriter writer = new BinaryWriter(stream);

            for (int layer = 0; layer < targetNetwork.Layers; layer++)
            {
                for (int neuron = 0; neuron < targetNetwork.Neurons[layer].Length; neuron++)
                {
                    double[] weights = targetNetwork.Neurons[layer][neuron].Weights;

                    for(int i = 0; i < weights.Length; i++)
                        writer.Write(weights[i]);

                    writer.Write(targetNetwork.Neurons[layer][neuron].Bias);
                }
            }

            writer.Close();
        }
    }

    class BackpropagationNetworkDeserializer
    {
        public BackpropagationNetwork targetNetwork;

        public BackpropagationNetworkDeserializer(BackpropagationNetwork TargetNetwork)
        {
            targetNetwork = TargetNetwork;
        }

        public void Deserialize(Stream stream)
        {
            BinaryReader reader = new BinaryReader(stream);

            for (int layer = 0; layer < targetNetwork.Layers; layer++)
            {
                for (int neuron = 0; neuron < targetNetwork.Neurons[layer].Length; neuron++)
                {
                    double[] weights = targetNetwork.Neurons[layer][neuron].Weights;

                    for (int i = 0; i < weights.Length; i++)
                        weights[i] = reader.ReadDouble();

                    targetNetwork.Neurons[layer][neuron].Bias = reader.ReadDouble();
                }
            }

            reader.Close();
        }
    }
}

using System;
using System.Collections.Generic;
using System.Text;

namespace NeuronCS.Algorithm
{
    public class GradientDecent
    {
        public readonly double GlobalLearningRate;

        public GradientDecent(double LearningRate)
        {
            GlobalLearningRate = LearningRate;
        }

        public virtual void Update(ref double[] Weights, double[] Gradient, ref double[] Summary, object OtherArguments, ref object OtherReference)
        {
            // Check parameters
            if (Weights.Length != Gradient.Length) throw new ArgumentException("The gradient size is different from weights tensor.", $"{nameof(Weights)} & {nameof(Gradient)}");

            int count = Weights.Length;

            for (int i = 0; i < count; i++)
            {
                // Update the weights
                Weights[i] -= GlobalLearningRate * Gradient[i];
            }
        }
    }

    public class AdaGradientDecent : GradientDecent
    {
        public readonly double MinimunDelta;

        public AdaGradientDecent(double GlobalLearningRate, double MinimunDelta = 1e-8) : base(GlobalLearningRate)
        {
            this.MinimunDelta = MinimunDelta;
        }

        public override void Update(ref double[] Weights, double[] Gradient, ref double[] Summary, object OtherArguments, ref object OtherReference)
        {
            // Check parameters
            if (Weights.Length != Gradient.Length) throw new ArgumentException("The gradient size is different from weights tensor.", $"{nameof(Weights)} & {nameof(Gradient)}");
            if (Gradient.Length != Summary.Length) throw new ArgumentException("The gradient size is different from summary size.", $"{nameof(Gradient)} & {nameof(Summary)}");

            int count = Weights.Length;

            for (int i = 0; i < count; i++)
            {
                // Cumulative squared gradient
                Summary[i] += Gradient[i] * Gradient[i];
                
                // Dynamic update learning rate
                double dynamicRate = GlobalLearningRate / (Math.Sqrt(Summary[i] + MinimunDelta));

                // Update the weights
                Weights[i] -= dynamicRate * Gradient[i];
            }
        }
    }

    public class AdaDeltaDecent : GradientDecent
    {
        public readonly double MeanFactor;
        public readonly double MinimunDelta;

        public AdaDeltaDecent(double GlobalLearningRate, double MeanFactor = 0.9, double MinimunDelta = 1) : base(GlobalLearningRate)
        {
            if ((MeanFactor < 0) || (MeanFactor > 1)) throw new ArgumentException("The mean factor must be 0 ~ 1", nameof(MeanFactor));

            this.MeanFactor = MeanFactor;
            this.MinimunDelta = MinimunDelta;
        }

        public override void Update(ref double[] Weights, double[] Gradient, ref double[] Summary, object OtherArguments, ref object OtherReference)
        {
            // Other arguments : previous gradient : double[]
            // Check parameters
            if (Weights.Length != Gradient.Length) throw new ArgumentException("The gradient size is different from weights tensor.", $"{nameof(Weights)} & {nameof(Gradient)}");
            if (Gradient.Length != Summary.Length) throw new ArgumentException("The gradient size is different from summary size.", $"{nameof(Gradient)} & {nameof(Summary)}");

            double[] Previous;

            try
            {
                Previous = OtherReference as double[];
            }
            catch
            {
                throw new ArgumentException($"The {nameof(OtherReference)} is previous gradient and it will be a array of double.", nameof(OtherReference));
            }


            int count = Weights.Length;

            for (int i = 0; i < count; i++)
            {
                // Cumulative squared gradient
                // Update summary
                Summary[i] =  MeanFactor * Previous[i] + (1 - MeanFactor) * Gradient[i] * Gradient[i];

                // Dynamic update learning rate
                double dynamicRate = GlobalLearningRate / (Math.Sqrt(Summary[i] + MinimunDelta));

                // Update the weights
                Weights[i] -= dynamicRate * Gradient[i];

                // Recoard previous gradient
                Previous[i] = Gradient[i] * Gradient[i];
            }
        }
    }

    public class AdamDecent : GradientDecent
    {
        public readonly double Mu;
        public readonly double Upsilon;
        public readonly double MinimunDelta;

        private double MuMul;
        private double UpsilonMul;

        public AdamDecent(double GlobalLearningRate = 0.001, double Mu = 0.9, double Upsilon = 0.999, double MinimunDelta = 1e-8) : base(GlobalLearningRate)
        {
            if ((Mu < 0) || (Mu > 1)) throw new ArgumentException("The mu must be 0 ~ 1", nameof(Mu));
            if ((Upsilon < 0) || (Upsilon > 1)) throw new ArgumentException("The upsilon must be 0 ~ 1", nameof(Upsilon));

            this.Mu = Mu;
            this.Upsilon = Upsilon;
            this.MinimunDelta = MinimunDelta;

            ClearCorrectAccumulate();
        }

        public void ClearCorrectAccumulate()
        {
            MuMul = Mu;
            UpsilonMul = Upsilon;
        }

        public void CorrectAccumulate()
        {
            // Update the correction index
            MuMul *= Mu;
            UpsilonMul *= Upsilon;
        }

        public override void Update(ref double[] Weights, double[] Gradient, ref double[] Summary, object OtherArguments, ref object OtherReference)
        {
            // Other arguments : previous gradient : double[]
            // Check parameters
            if (Weights.Length != Gradient.Length) throw new ArgumentException("The gradient size is different from weights tensor.", $"{nameof(Weights)} & {nameof(Gradient)}");

            List<double[]> ParametersRef;
            try
            {
                ParametersRef = OtherReference as List<double[]>;
            }
            catch
            {
                throw new ArgumentException($"The {nameof(OtherReference)} is previous gradient and it will be a list of array of double.", nameof(OtherReference));
            }

            double[] prevM = ParametersRef[0];
            double[] prevN = ParametersRef[1];

            if (prevM.Length != Gradient.Length) throw new ArgumentException("The gradient size is different from M tensor.", $"{nameof(Gradient)} & {nameof(OtherReference)}[0]");
            if (prevN.Length != Gradient.Length) throw new ArgumentException("The gradient size is different from N tensor.", $"{nameof(Gradient)} & {nameof(OtherReference)}[1]");

            int count = Weights.Length;

            for (int i = 0; i < count; i++)
            {
                // Attenuation
                double M = Mu * prevM[i] + (1 - Mu) * Gradient[i];
                double N = Upsilon * prevN[i] + (1 - Upsilon) * Gradient[i] * Gradient[i];

                // Correction
                double Me = M / (1 - MuMul);
                double Ne = N / (1 - UpsilonMul);

                // Gradient decent
                Weights[i] -= Me / (Math.Sqrt(Ne) + MinimunDelta) * GlobalLearningRate;

                // Update previous M and N
                prevM[i] = M;
                prevN[i] = N;
            }

            List<double[]> NewParameters = new List<double[]>()
            {
                prevM, prevN
            };

            OtherReference = NewParameters;
        }
    }
}

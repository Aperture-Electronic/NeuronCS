using NeuronCS.Types;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace NeuronCS.Functions
{
    /// <summary>
    /// Invalid double array function, always O = 0 whatever the input is
    /// </summary>
    public class InvalidDoubleArrayFunction : DoubleArrayFunction
    {
        public InvalidDoubleArrayFunction() :
            base((double[] x, double[] y) => (0), (double[] x, double[] y, double o, int p) => (0), (double[] x, double[] y, double o, int p) => (0))
        {

        }
    }

    /// <summary>
    /// Weighted summation double array function, O = Σ(Ai * Bi)
    /// </summary>
    public class SumDoubleArrayFunction : DoubleArrayFunction
    {
        public SumDoubleArrayFunction() : 
            base(delegate (double[] i, double[] w)
                    {
                        int length = i.Length;
                        double[] k = new double[length];

                        k = i.Zip(w, (double x, double e) => (x * e)).ToArray();
                        return k.Sum();
                    }, 
                    delegate (double[] i, double[] w, double o, int index)
                    {
                        return w[index];
                    }, 
                    delegate (double[] i, double[] w, double o, int index)
                    {
                        return i[index];
                    })
        {

        }
    }

    public class MeanSquareError : DoubleArrayFunction
    {
        public MeanSquareError() : 
            base(delegate(double[] current, double[] target)
            {
                IEnumerable<double> ms = current.Zip(target, (double c, double t) => ((t - c) * (t - c)));
                return 0.5 * ms.Sum();
            }, (double[] current, double[] target, double d, int i) => ((current[i] + target[i]) * d), (double[] current, double[] target, double d, int i) => ((target[i] - current[i]) * d))
        {

        }
    }

    /// <summary>
    /// Weighted average double array function, O = (1/N) * Σ(Ai * Bi)
    /// </summary>
    public class AverageDoubleArrayFunction : DoubleArrayFunction
    {
        public AverageDoubleArrayFunction() :
            base(delegate (double[] i, double[] w)
            {
                int length = i.Length;
                double[] k = new double[length];

                k = i.Zip(w, (double x, double e) => (x * e)).ToArray();
                return k.Average();
            },
                    delegate (double[] i, double[] w, double o, int index)
                    {
                        return w[index] / i.Length;
                    },
                    delegate (double[] i, double[] w, double o, int index)
                    {
                        return i[index] / i.Length;
                    })
        {

        }
    }

    /// <summary>
    /// Step function, when I > 0, O = 1, else O = 0
    /// </summary>
    public class StepFunction : SingleValueFunction
    {
        public StepFunction() : base((double x) => ((x  <= 0) ? (0) : (1)), (double d) => ((d != 0) ? (0) : (double.NaN)))
        {

        }
    }

    /// <summary>
    /// Sign function, when I > 0, O = 1, else O = -1 
    /// </summary>
    public class SignFunction : SingleValueFunction
    {
        public SignFunction() : base((double x) => ((x <= 0) ? (-1) : (1)), (double d) => ((d != 0) ? (0) : (double.NaN)))
        {

        }
    }

    /// <summary>
    /// Linear function, O = I
    /// </summary>
    public class LinearFunction : SingleValueFunction
    {
        public LinearFunction() : base((double x) => (x), (double d) => (1))
        {

        }
    }

    /// <summary>
    /// Ramp function, when I ∈ (0, 1), O = I, else O = 0
    /// </summary>
    public class RampFunction : SingleValueFunction
    {
        public RampFunction() : base((double x) => ((x < 0) ? (0) : ((x > 1) ? (1) : (x))), (double d) => (((d < 0) || (d > 1)) ? (0) : (1)))
        {

        }
    }

    /// <summary>
    /// Sigmoid function, O = 1 / (1 + Exp(-I))
    /// </summary>
    public class SigmoidFunction : SingleValueFunction
    {
        public SigmoidFunction() : base(delegate(double x)
        {
            double exp = Math.Exp(-x);
            if (double.IsInfinity(exp)) return 1;
            else return (1.0 / (1.0 + exp));
        },
            delegate(double d)
            {
                double ed = 1.0 + Math.Exp(d);
                if (double.IsInfinity(ed)) return 0;
                return (ed - 1.0) / (ed * ed);
            })
        {

        }
    }

    /// <summary>
    /// Hyperbolic tangent function (Tanh), O = Tanh(I)
    /// </summary>
    public class HyperbolicTangentFunction : SingleValueFunction
    {
        public HyperbolicTangentFunction() : base((double x) => (Math.Tanh(x)), (double d) => (2 / (1 + Math.Cosh(2 * d))))
        {

        }
    }

    /// <summary>
    /// Rectified linear unit function (ReLU), when I > 0, O = I, else O = 0
    /// </summary>
    public class ReLUFunction : SingleValueFunction
    {
        public ReLUFunction() : base((double x) => ((x < 0) ? (0) : (x)), (double d) => ((d < 0) ? (0) : (1)))
        {

        }
    }
}

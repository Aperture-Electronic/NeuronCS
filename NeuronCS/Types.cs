using System;
using System.Collections.Generic;
using System.Text;

namespace NeuronCS.Types
{
    public class DoubleArrayFunction
    {
        public Func<double[], double[], double> Function;
        public Func<double[], double[], double, int, double> FirstParaDerivatives;
        public Func<double[], double[], double, int, double> SecondParaDerivatives;

        public DoubleArrayFunction(Func<double[], double[], double> Function, Func<double[], double[], double, int, double> FirstParaDerivatives, Func<double[], double[], double, int, double> SecondParaDerivatives)
        {
            this.Function = Function;
            this.FirstParaDerivatives = FirstParaDerivatives;
            this.SecondParaDerivatives = SecondParaDerivatives;
        }
    }

    public class SingleValueFunction
    {
        public Func<double, double> Function;
        public Func<double, double> Derivatives;

        public SingleValueFunction(Func<double, double> Function, Func<double, double> Derivatives)
        {
            this.Function = Function;
            this.Derivatives = Derivatives;
        }
    }
}

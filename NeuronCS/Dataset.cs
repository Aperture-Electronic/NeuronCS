using System;
using System.Collections;
using System.Collections.Generic;
using System.Text;

namespace NeuronCS.Dataset
{

    public struct IOMetaDataSetItem<T>
    {
        public T DataIn;
        public T DataOut;
    }

    public class IOMetaDataSet<T> : IEnumerable<IOMetaDataSetItem<T>> where T : IEnumerable
    {
        private List<T> DataSetIn;
        private List<T> DataSetOut;

        public int Size => DataSetIn.Count;

        public IOMetaDataSetItem<T> this[int index]
        {
            get { return new IOMetaDataSetItem<T>() { DataIn = DataSetIn[index], DataOut = DataSetOut[index] }; }
            set { DataSetIn[index] = value.DataIn; DataSetOut[index] = value.DataOut; }
        }

        public IOMetaDataSet(List<T> DataSetIn, List<T> DataSetOut)
        {
            // Check parameters
            if (DataSetIn.Count != DataSetOut.Count) throw new ArgumentException("Two meta data set size is not matching.", $"{nameof(DataSetIn)} & {nameof(DataSetOut)}");

            this.DataSetIn = DataSetIn;
            this.DataSetOut = DataSetOut;
        }

        public IEnumerator<IOMetaDataSetItem<T>> GetEnumerator()
        {
            for (int i = 0; i < Size; i++)
            {
                yield return new IOMetaDataSetItem<T>() { DataIn = DataSetIn[i], DataOut = DataSetOut[i] };
            }
        }

        IEnumerator IEnumerable.GetEnumerator()
        {
            for (int i = 0; i < Size; i++)
            {
                yield return new IOMetaDataSetItem<T>() { DataIn = DataSetIn[i], DataOut = DataSetOut[i] };
            }
        }
    }
}

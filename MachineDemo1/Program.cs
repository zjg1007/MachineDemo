using Microsoft.ML;
using Microsoft.ML.Runtime.Api;
using Microsoft.ML.Trainers;
using Microsoft.ML.Transforms;
using System;

namespace MachineDemo1
{
    class Program
    {
     
        //步骤1:定义数据结构。
        // IrisData用于提供培训数据，以及as。
        //预测操作输入。
        // -前4个属性是用来预测标签的输入/特性。
        // -标签是你所预测的，只有在训练时才设置。
        public class IrisData
        {
            [Column("0")]
            public float SepalLength;

            [Column("1")]
            public float SepalWidth;

            [Column("2")]
            public float PetalLength;

            [Column("3")]
            public float PetalWidth;

            [Column("4")]
            [ColumnName("Label")]
            public string Label;
        }
        //IrisPrediction是预测操作返回的结果。
        public class IrisPrediction
        {
            [ColumnName("PredictedLabel")]
            public string PredictedLabels;
        }


        static void Main(string[] args)
        {
            //步骤2:创建管道并加载数据。
            var pipeline = new LearningPipeline();

            //如果在Visual Studio中工作，请确保“复制到输出目录”
            // 虹膜数据的属性.txt被设置为'Copy always'
            string dataPath = "iris-data.txt";
            pipeline.Add(new TextLoader<IrisData>(dataPath, separator: ","));

            //第三步:转换你的数据。
            //将数值赋值给“Label”列中的文本，因为只有这样。
            //数字可以在模型训练期间进行处理。
            pipeline.Add(new Dictionarizer("Label"));

            //将所有的特性都放到一个向量中。
            pipeline.Add(new ColumnConcatenator("Features", "SepalLength", "SepalWidth", "PetalLength", "PetalWidth"));

            //第四步:增加学习者。
            //将学习算法添加到管道中。
            //这是一个分类场景(这是什么类型的iris ?)
            pipeline.Add(new StochasticDualCoordinateAscentClassifier());

            //将标签转换为原始文本(在步骤3中转换为数字后)
            pipeline.Add(new PredictedLabelColumnOriginalValueConverter() { PredictedLabelColumn = "PredictedLabel" });

            //步骤5:根据数据集训练你的模型。
            var model = pipeline.Train<IrisData, IrisPrediction>();

            //第六步:用你的模型做一个预测。
            //你可以改变这些数字来测试不同的预测。
            var prediction = model.Predict(new IrisData()
            {
                SepalLength = 3.3f,
                SepalWidth = 1.6f,
                PetalLength = 0.2f,
                PetalWidth = 5.1f,
            });

            Console.WriteLine($"Predicted flower type is: {prediction.PredictedLabels}");
        }
    }
}

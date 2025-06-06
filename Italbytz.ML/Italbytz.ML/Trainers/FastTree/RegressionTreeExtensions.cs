using System.Globalization;
using System.Text;
using Microsoft.ML.Trainers.FastTree;

namespace Italbytz.ML.Trainers.FastTree;

/// <summary>
///     Extension methods for working with <see cref="RegressionTree" /> instances.
/// </summary>
public static class RegressionTreeExtensions
{
    /// <summary>
    ///     Converts a regression tree to a Graphviz DOT representation.
    /// </summary>
    /// <param name="tree">The regression tree to convert.</param>
    /// <returns>A string containing the Graphviz DOT representation of the tree.</returns>
    /// <remarks>
    ///     The generated Graphviz diagram uses boxes for leaf nodes (displaying leaf
    ///     values),
    ///     and plain nodes for decision nodes (displaying feature index and
    ///     threshold).
    ///     Edges are labeled with "≤" for the left child (values less than or equal to
    ///     threshold)
    ///     and ">" for the right child (values greater than threshold).
    /// </remarks>
    public static string ToGraphviz(
        this RegressionTree tree)
    {
        var sb = new StringBuilder();
        sb.AppendLine("digraph G {");
        sb.AppendLine("    rankdir=\"TB\"");
        for (var i = 0; i < tree.LeafValues.Count; i++)
        {
            var leafValue = tree.LeafValues[i]
                .ToString("F2", CultureInfo.InvariantCulture);
            sb.AppendLine(
                $"    l{i} [shape=box,label={leafValue}];");
        }

        for (var i = 0; i < tree.NumericalSplitFeatureIndexes.Count; i++)
        {
            var featureIndex = tree.NumericalSplitFeatureIndexes[i];
            var threshold = tree.NumericalSplitThresholds[i]
                .ToString("F2", CultureInfo.InvariantCulture);
            sb.AppendLine(
                $"    n{i} [shape=plain,label=<Feature{featureIndex}<br/>{threshold}>];");
        }

        for (var i = 0; i < tree.LeftChild.Count; i++)
        {
            var leftChildType = tree.LeftChild[i] < 0 ? "l" : "n";
            var leftChildIndex = tree.LeftChild[i] < 0
                ? ~tree.LeftChild[i]
                : tree.LeftChild[i];
            var rightChildType = tree.RightChild[i] < 0 ? "l" : "n";
            var rightChildIndex = tree.RightChild[i] < 0
                ? ~tree.RightChild[i]
                : tree.RightChild[i];
            var leftChild = leftChildType + leftChildIndex;
            var rightChild = rightChildType + rightChildIndex;
            sb.AppendLine($"    n{i} -> {leftChild} [label=\"≤\"];");
            sb.AppendLine($"    n{i} -> {rightChild} [label=\">\";]");
        }

        sb.AppendLine("}");
        return sb.ToString();
    }

    /// <summary>
    ///     Converts a regression tree to a PlantUML string representation.
    /// </summary>
    /// <param name="tree">The regression tree to convert.</param>
    /// <returns>
    ///     A string containing the PlantUML representation of the regression
    ///     tree.
    /// </returns>
    /// <remarks>
    ///     The PlantUML representation includes the following tree properties:
    ///     - NumericalSplitFeatureIndexes: The feature indexes used for numerical
    ///     splits
    ///     - NumericalSplitThresholds: The threshold values for numerical splits
    ///     (formatted to 2 decimal places)
    ///     - LeftChild: The indices of left child nodes
    ///     - RightChild: The indices of right child nodes
    ///     - LeafValues: The values at leaf nodes (formatted to 2 decimal places)
    /// </remarks>
    public static string ToPlantUML(
        this RegressionTree tree)
    {
        var sb = new StringBuilder();
        sb.AppendLine("@startuml");
        sb.AppendLine("object RegressionTree {");
        sb.AppendLine(
            $"    int[] NumericalSplitFeatureIndexes = [{string.Join(", ", tree.NumericalSplitFeatureIndexes)}]");
        var numericalSplitThresholdsStrings = tree.NumericalSplitThresholds
            .Select(v => v.ToString("F2", CultureInfo.InvariantCulture))
            .ToArray();
        sb.AppendLine(
            $"    double[] NumericalSplitThresholds = [{string.Join(", ", numericalSplitThresholdsStrings)}]");
        sb.AppendLine(
            $"    int[] LeftChild = [{string.Join(", ", tree.LeftChild)}]");
        sb.AppendLine(
            $"    int[] RightChild = [{string.Join(", ", tree.RightChild)}]");
        var leafValueStrings = tree.LeafValues
            .Select(v => v.ToString("F2", CultureInfo.InvariantCulture))
            .ToArray();
        sb.AppendLine(
            $"    double[] LeafValues = [{string.Join(", ", leafValueStrings)}]");
        sb.AppendLine("}");
        sb.AppendLine("@enduml");
        return sb.ToString();
    }
}
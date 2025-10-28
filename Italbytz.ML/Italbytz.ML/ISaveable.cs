namespace Italbytz.ML;

/// <summary>
///     Defines a contract for types that can be serialized to a stream.
///     Implementations should write sufficient data to reconstruct the object.
/// </summary>
public interface ISaveable
{
    /// <summary>
    ///     Serializes the object to the provided stream. Implementations should
    ///     write sufficient data to reconstruct the object.
    /// </summary>
    /// <param name="stream">The target stream to which the object is saved.</param>
    void Save(Stream stream);
}
namespace Italbytz.ML;

/// <summary>
///     Defines a loader that can deserialize an instance of
///     <typeparamref name="TClassToLoad" /> from a stream.
/// </summary>
/// <typeparam name="TClassToLoad">The type of object to load from the stream.</typeparam>
public interface ICanLoad<TClassToLoad>
{
    /// <summary>
    ///     Loads an instance of <typeparamref name="TClassToLoad" /> from the provided
    ///     <see cref="System.IO.Stream" />.
    ///     The caller remains responsible for disposing the stream if required.
    /// </summary>
    /// <param name="stream">
    ///     Stream containing the serialized representation of the
    ///     object.
    /// </param>
    /// <returns>
    ///     An instance of <typeparamref name="TClassToLoad" /> deserialized from
    ///     the stream.
    /// </returns>
    TClassToLoad Load(Stream stream);
}
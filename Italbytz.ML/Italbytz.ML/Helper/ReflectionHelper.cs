using System.Reflection;

namespace Italbytz.ML.Helper;

public class ReflectionHelper
{
    public static PropertyInfo[] GetPropertyInfo<T>(T obj, BindingFlags flags)
    {
        return typeof(T).GetProperties(flags);
    }

    public static IEnumerable<(string name, object value)> GetPropertyValues<T>(
        T obj, BindingFlags flags)
    {
        return from p in typeof(T).GetProperties(flags)
            where p.GetIndexParameters().Length == 0 //To filter out indexers
            select (p.Name, p.GetValue(obj, null));
    }

    public static IEnumerable<(string name, object value)> GetFieldValues<T>(
        T obj, BindingFlags flags)
    {
        return typeof(T).GetFields(flags)
            .Select(f => (f.Name, f.GetValue(obj)));
    }
}
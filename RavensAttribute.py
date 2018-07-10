# A single variable-value pair that describes some element of a RavensObject.
# For example, a circle might have the attributes shape:circle, size:large, and
# filled:no.

class RavensAttribute:
    # Creates a new RavensAttribute.
    #
    # Your agent does not need to use this method.
    #
    # @param name the name of the attribute
    # @param value the value of the attribute
    def __init__(self, name, value):
        self.name=name
        self.value=value

    # Returns the name of the attribute. For example, 'shape', 'size', or
    # 'fill'.
    #
    # @return the name of the attribute
    def getName(self):
        return self.name

    # Returns the value of the attribute. For example, 'circle', 'large', or
    # 'no'.
    #
    # @return the value of the attribute
    def getValue(self):
        return self.value

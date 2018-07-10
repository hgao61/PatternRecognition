
# A single object in a RavensFigure -- typically, a single shape in a frame,
# such as a triangle or a circle -- comprised of a list of RavensAttributes.
class RavensObject:
    # Constructs a new RavensObject given a name.
    #
    # Your agent does not need to use this method.
    #
    # @param name the name of the object
    def __init__(self, name):
        self.name=name
        self.attributes=[]

    # The name of this RavensObject. Names are assigned starting with the
    # letter Z and proceeding backwards in the alphabet through the objects
    # in the Frame. Names do not imply correspondence between shapes in
    # different frames. Names are simply provided to allow agents to organize
    # their reasoning over different figures.
    #
    # Within a RavensFigure, each RavensObject has a unique name.
    #
    # @return the name of the RavensObject
    def getName(self):
        return self.name

    # Returns an ArrayList of RavensAttribute characterizing this RavensObject.
    #
    # @return an ArrayList of RavensAttribute
    def getAttributes(self):
        return self.attributes

import numpy as np
import gtda
from gtda.homology import VietorisRipsPersistence
from gtda.plotting import plot_diagram
from gtda.diagrams import PersistenceEntropy
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

class Space:
    """
    Create a space object with a name and an embedding dimension
    """
    def __init__(self, name, edim, dim=None):
        self.name = name
        self.edim = edim
        self.dim = dim if dim is not None else edim
        self.points = None

    def sample(self, sample_size, num_samples = 1):
        raise NotImplementedError()

    def poincare_polynomial(self):
        raise NotImplementedError()

    def fit(self, sample_size, num_samples = 1, homology_dimensions = None):
        x = self.sample(sample_size, num_samples)
        if homology_dimensions is None:
            homology_dimensions = list(range(self.dim+2))
        self.homology_dimensions = homology_dimensions
        VR = gtda.homology.VietorisRipsPersistence(homology_dimensions=homology_dimensions)
        self.VR = VR
        self.diagrams = VR.fit_transform(x)
        return self.diagrams
    
    def uncomplex (self, x):
        y = np.array([np.real(x),np.imag(x)])
        y = np.moveaxis(y,0,-1)
        y = y.reshape(y.shape[0],y.shape[1],2*y.shape[2])
        return y
        
class Sphere(Space):
    def __init__(self, dim):
        """
        Create a sphere of dimension dim.  This is the set of unit vectors 
        in R ^(dim + 1), so the embedding dimension is dim+1.
        """
        self.dim = dim
        super().__init__('S^{' + str(dim) + '}', dim+1, dim)

    def sample(self, sample_size, num_samples = 1):
        """
        Returns a numpy matrix of shape n x edim, where each row is a
        random point on the sphere.  To do this, we generate a matrix
        of shape n x edim of standard normal random variables, and then
        normalize each row to have unit length.
        """
        x = np.random.randn(sample_size * num_samples, self.edim)
        x /= np.linalg.norm(x, axis=1)[:, None]
        self.points = x.reshape(num_samples, sample_size, self.edim)
        return self.points

    def poincare_polynomial(self):
        """
        This returns the correct answer for the Poincare polynomial 
        of the sphere, represented as an integer-valued numpy array.
        The i-th entry is the coefficient of t^i, which is the rank
        of the i-th homology group.  In the case of the sphere, the
        Poincare polynomial is 1 + t^dim.
        """
        p = np.zeros(self.dim+1, dtype=int)
        p[0] = 1
        p[self.dim] = 1
        return p

class ComplexProjectiveSpace(Space):
    def __init__(self, cdim):
        self.cdim = cdim
        self.dim = 2*cdim
        super().__init__('CP^{' + str(cdim) + '}', 2*(cdim+1) ** 2, 2*self.cdim)

    def sample(self, sample_size, num_samples = 1):
        d = self.cdim+1
        n = sample_size * num_samples
        x = np.random.randn(n, d) + 1j * np.random.randn(n, d)
        x /= np.linalg.norm(x, axis=1)[:, None]
        x = x.reshape(n,d,1) * np.conj(x.reshape(n, 1, d))
        x = np.array([np.real(x), np.imag(x)]).reshape(n, 2*d*d)
        x = x.reshape(num_samples, sample_size, self.edim)
        x = self.uncomplex(x)
        self.points = x
        return self.points

    def poincare_polynomial(self):
        d = self.cdim
        p = np.zeros(2*d + 2, dtype=int)
        p[0::2] = 1
        return p
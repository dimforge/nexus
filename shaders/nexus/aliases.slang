#if DIM == 2
import gla.geometry.sim2;
import gla.geometry.rot2;

typealias Pose = Sim2;
typealias AngVect = float;
typealias Rot = Rot2;
#else
import gla.geometry.sim3;
import gla.geometry.quat;

typealias Pose = Sim3;
typealias AngVect = float3;
typealias Rot = Quat;
#endif

typealias Vect<T = float> = vector<T, DIM>;
typealias Mat<T = float> = matrix<T, DIM, DIM>;
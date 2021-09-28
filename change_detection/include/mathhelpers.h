#ifndef MATHHELPERS_H
#define MATHHELPERS_H

#include "pcl/ModelCoefficients.h"


struct PlaneStruct {
    pcl::PointIndices::Ptr plane_ind;
    pcl::ModelCoefficients::Ptr coeffs;
    float avg_z;

    PlaneStruct(){}
    PlaneStruct(pcl::PointIndices::Ptr ind, pcl::ModelCoefficients::Ptr c, float z) : plane_ind(ind), coeffs(c), avg_z(z) {}

    bool operator < (const PlaneStruct& plane) const
    {
        //std::cout<<avg_z << " " << plane.avg_z << std::endl;
        //return (-coeffs->values[3]/coeffs->values[2] < -plane.coeffs->values[3]/plane.coeffs->values[2]);
        return avg_z < plane.avg_z;
    }

    PlaneStruct& operator=(const PlaneStruct rhs) {
        plane_ind = boost::make_shared<pcl::PointIndices>(*(rhs.plane_ind));
        avg_z =  rhs.avg_z;
        coeffs = boost::make_shared<pcl::ModelCoefficients>(*(rhs.coeffs));
        return *this;
    }
};

struct PlaneWithObjInd {
    PlaneStruct plane;
    std::vector<int> obj_indices;
};

#endif // MATHHELPERS_H

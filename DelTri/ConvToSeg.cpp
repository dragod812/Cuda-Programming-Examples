

#include "ConvToSeg.h"
//#include "global_datatype.h"
//funtion to Extend/Limit the rays to the Bounding Box(The first box in the QuadTree)
Segment_2 convToSeg(const Iso_rectangle_2& box, const Ray_2& ray){
	//std::cout.precision(17);
	CGAL::Object obj = CGAL::intersection(ray, box);
	//std::cout<<ray<<" Ray in convtoseg"<<std::endl;

	const Point_2* tempPoint = CGAL::object_cast<Point_2>(&obj);
	const Segment_2* tempSeg = CGAL::object_cast<Segment_2>(&obj);
	//Segment_2 seg;

	if (tempPoint != nullptr){
		//std::cout<<" In point convtoseg"<<std::endl;
		Segment_2 temp(ray.source(), *tempPoint);
		//seg=temp;
		return temp;
	}
	if (tempSeg != nullptr){
		//std::cout<<" In segment convtoseg"<<std::endl;
		//seg=*tempSeg;
		return *tempSeg;
	}

	//std::cout<<seg<<" Ray in convtoseg"<<std::endl;
}
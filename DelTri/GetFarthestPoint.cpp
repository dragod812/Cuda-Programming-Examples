

#include "GetFarthestPoint.h"

//Function to get the farthest point
//template <typename Type1,typename Type2,typename Type3>
Point_2 get_Farthest_Point(Segment_2& ray, std::vector<Segment_2>& seg, Point_2& point)
{
	//std::cout.precision(17);
	//std::cout<<"Error get Farthest Point"<<std::endl;
	Point_2 Point;
	double d_max = 0;
	for (int i = 0; i<seg.size(); i++)
	{
		//std::cout<<"Error get Farthest Point 2"<<std::endl;
		//std::vector<Point_2> tempVec;
		Point_2 far_point;
		if ((CGAL::do_intersect(ray, seg.at(i))) && (seg.at(i).source() != seg.at(i).target()))
		{

			const Point_2* tempPoint;
			if (seg.at(i).is_vertical()){

				//std::cout<<" IsVertical segment is vertical"<<std::endl;
				double t = ((seg.at(i).source().x() - ray.source().x()) / ray.direction().dx());
				Point_2 pq(seg.at(i).source().x(), ray.source().y() + t*(ray.direction().dy()));
				far_point = pq;
			}
			else{
				CGAL::Object obj = CGAL::intersection(ray, seg.at(i));
				tempPoint = CGAL::object_cast<Point_2>(&obj);
				if (tempPoint == NULL){ std::cout << "tempPoint is NULL " << std::endl; std::cout << seg.at(i) << std::endl; }
				const Segment_2* tempSeg = CGAL::object_cast<Segment_2>(&obj);
				if (tempPoint){
					far_point = *tempPoint;//std::cout<<"It is a point"<<std::endl;
				}
				if (tempSeg){ std::cout << "It is a segment" << std::endl; }
				/*if(mulInter.is_open())
				{
				std::cout<<"Error get Farthest Point 4"<<std::endl;
				//std::cout<<"open"<<std::endl;
				mulInter<<*tempPoint<<std::endl;
				}*/

			}
			double dist = (CGAL::squared_distance(point, far_point));
			//std::cout<<"Error get Farthest Point 8"<<std::endl;
			if (dist>d_max)
			{
				//std::cout<<"Error get Farthest Point 9"<<std::endl;
				Point = far_point;
				d_max = dist;
			}

		}
		else{//std::cout<<"It is not intersecting"<<std::endl;
		}
	}
	return Point;

}
#include "InsidePoints.h"

bool ClipTest(double p, double q, double *u1, double *u2){
	double r; bool retVal = true;

	if (p < 0.0)
	{
		r = q / p;
		if (r > *u2)
			retVal = false;
		else if (r > *u1)
			*u1 = r;
	}
	else if (p > 0.0)
	{
		r = q / p;
		if (r < *u1)
			retVal = false;
		else if (r < *u2)
			*u2 = r;
	}
	else if (q < 0.0)
		retVal = false;
	return retVal;
}

bool intersect(const Segment_2& ray, const Iso_rectangle_2& box){
	double dx = ray.end().x() - ray.source().x(), dy, u1 = 0.0, u2 = 1.0, q[4];
	q[0] = ray.source().x() - box.xmin();
	q[1] = box.xmax() - ray.source().x();
	q[2] = ray.source().y() - box.ymin();
	q[3] = box.ymax() - ray.source().y();

	if (ClipTest(-dx, q[0], &u1, &u2))
	{
		if (ClipTest(dx, q[1], &u1, &u2))
		{
			double dy = ray.end().y() - ray.source().y();
			if (ClipTest(-dy, q[2], &u1, &u2))
			if (ClipTest(dy, q[3], &u1, &u2))
				//if(u2 < 1.0)
				//	return true;
				//if(u1 > 0.0)
				return true;

		}
	}
	else
		return false;

}

//Funtion to get the points lying inside the slab of spcified thickness/threshold and store it in an array 
template <typename Type1, typename Type2>
void thresh_Points(const Type1& Object1, const Type2& Object2, const double& threshold, std::vector<Type2>& Object3)
{
	//std::cout.precision(17);
	if (CGAL::squared_distance(Object1, Object2) <= threshold)
	{
		//std::cout<<"C"<<std::endl;
		Object3.push_back(Object2);
		//std::cout<<"2.1"<<std::endl;
	}
}

//funtion to get a bounding box given a ray(converted to a segment) or a segment for a given threshold value
std::vector<Segment_2> bBox(const Segment_2& edge, const double& thresh)
{
	//std::cout.precision(17);
	double dist = std::sqrt(thresh);
	//dist=std::sqrt(dist);
	//std::cout<<"inside bBox"<<std::endl;
	//Segment_2 tempSeg(Point_2(0,0),Point_2(0,0));

	//std::cout<<edge.direction().dx()<<"  "<<edge.direction().dy()<<std::endl;

	Direction_2 dir = edge.direction();
	//std::cout<<"inside bBox"<<std::endl;
	//std::cout<<"dir.dx	"<<dir.dx()<<std::endl;
	if (dir.dx() == 0)
	{
		Point_2 bottomL(edge.source().x(), edge.source().y() - dist);
		Point_2 bottomR(edge.source().x(), edge.source().y() + dist);
		Point_2 topR(edge.target().x(), edge.target().y() + dist);
		Point_2 topL(edge.target().x(), edge.target().y() - dist);

		Segment_2 temp(bottomL, topL);
		Segment_2 temp1(bottomR, topR);
		std::vector<Segment_2> tempVec;
		//tempVec.reserve(3);
		tempVec.push_back(temp);
		tempVec.push_back(temp1);
		tempVec.push_back(edge);

		return tempVec;
	}
	else
	{
		double slope = (dir.dy() / dir.dx());

		Point_2 bottomL(edge.source().x() - dist*std::cos(std::atan(-1 / slope)), edge.source().y() - dist*std::sin(std::atan(-1 / slope)));
		Point_2 topR(edge.target().x() + dist*std::cos(std::atan(-1 / slope)), edge.target().y() + dist*std::sin(std::atan(-1 / slope)));
		Point_2 bottomR(edge.source().x() + dist*std::cos(std::atan(-1 / slope)), edge.source().y() + dist*std::sin(std::atan(-1 / slope)));
		Point_2 topL(edge.target().x() - dist*std::cos(std::atan(-1 / slope)), edge.target().y() - dist*std::sin(std::atan(-1 / slope)));

		Segment_2 temp(bottomL, topL);
		Segment_2 temp1(bottomR, topR);
		std::vector<Segment_2> tempVec;
		//tempVec.reserve(3);
		tempVec.push_back(temp);
		tempVec.push_back(temp1);
		tempVec.push_back(edge);

		return tempVec;
	}

}

std::vector<Point_2> insidePoints(const QuadTree& tree, const Segment_2& edge, const double& thresh)
{
	//std::cout.precision(17);
	std::vector<std::vector<Point_2> > tempPoints;
	//tempPoints.reserve(5);
	std::queue<std::shared_ptr<Node>> nodeQueue;
	//std::unordered_map<Point_2,int> mapping;
	//int count=0;

	std::vector<Point_2> th_Points;
	//th_Points.reserve(10);
	std::shared_ptr<Node> temp;
	//temp.reserve(4);

	//std::cout<<"insidePoint"<<std::endl;
	std::vector<Segment_2> tempSeg = bBox(edge, thresh);
	//std::cout<<"bBox"<<std::endl;

	if (tree.rootNode == nullptr){ return th_Points; }
	else
	{
		nodeQueue.push(tree.rootNode);
		while (!nodeQueue.empty()){
			temp = nodeQueue.front();

			if (temp->child.size() == 0)
			{
				tempPoints.push_back(temp->insidePoints);
			}

			for (int i = 0; i<temp->child.size(); ++i)
			{
				bool boxInside = false;
				if ((CGAL::squared_distance(tempSeg.at(2), temp->child.at(i)->rectangle.vertex(0)) <= thresh) || (CGAL::squared_distance(tempSeg.at(2), temp->child.at(i)->rectangle.vertex(1)) <= thresh) || (CGAL::squared_distance(tempSeg.at(2), temp->child.at(i)->rectangle.vertex(2)) <= thresh) || (CGAL::squared_distance(tempSeg.at(2), temp->child.at(i)->rectangle.vertex(3)) <= thresh))
				{
					boxInside = true;
				}

				//if((boxInside==true) || (CGAL::do_intersect(tempSeg.at(0),temp->child.at(i)->rectangle))||(CGAL::do_intersect(tempSeg.at(1),temp->child.at(i)->rectangle))||(CGAL::do_intersect(tempSeg.at(2),temp->child.at(i)->rectangle)))
				if ((boxInside == true) || (intersect(tempSeg.at(0), temp->child.at(i)->rectangle)) || (intersect(tempSeg.at(1), temp->child.at(i)->rectangle)) || (intersect(tempSeg.at(2), temp->child.at(i)->rectangle)))
				{
					nodeQueue.push(temp->child.at(i));
				}
			}
			nodeQueue.pop();
		}
	}
	for (int i = 0; i<tempPoints.size(); ++i){
		//std::cout<<"2"<<std::endl;
		for (int j = 0; j<tempPoints[i].size(); ++j){
			thresh_Points(edge, tempPoints[i].at(j), thresh, th_Points);
		}
	}
	//std::vector<Point_2> th1_Points;
	//th1_Points.reserve(10);
	//std::cout<<"done"<<std::endl;
	return th_Points;
}

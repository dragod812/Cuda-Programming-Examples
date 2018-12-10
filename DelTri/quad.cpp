
#include "global_datatype.h"

void quad(){
class Node
{
public:
	Node();
	//Node(float xmin,float ymin,float xmax,float ymax);
	Node(const Point_2& bottom, const Point_2& top);
	Node(const Iso_rectangle_2& box);
	virtual ~Node(){}

	std::vector<Point_2> insidePoints;
	Iso_rectangle_2 rectangle;
	Point_2 minCoor, maxCoor;

	std::shared_ptr<Node> parent;
	std::vector<std::shared_ptr<Node> > child;

};

Node::Node() :minCoor(0, 0), maxCoor(0, 0), rectangle(minCoor, maxCoor){}
Node::Node(const Point_2& bottom, const Point_2& top) : minCoor(bottom), maxCoor(top), rectangle(bottom, top){}
Node::Node(const Iso_rectangle_2& box) : rectangle(box), minCoor(box.min()), maxCoor(box.max()){}

class QuadTree
{
public:
	QuadTree();
	QuadTree(const Point_2& bottom, const Point_2& top);
	QuadTree(const Iso_rectangle_2& box);
	virtual ~QuadTree(){}

	//bool is_leafNode();
	bool is_leafNode(const std::shared_ptr<Node>& node);
	void spilt(std::shared_ptr<Node>& node);
	void insertPoints(std::shared_ptr<Node>& node);
	int checkNumPoints(const Node& node);

	//Point_2 leftDown,rightUp;
	std::shared_ptr<Node> rootNode;
	std::shared_ptr<Node> leafNode;
};

QuadTree::QuadTree(){}
//QuadTree::~QuadTree(){}
QuadTree::QuadTree(const Point_2& bottom, const Point_2& top){ rootNode->maxCoor = top; rootNode->minCoor = Point_2(bottom); }
QuadTree::QuadTree(const Iso_rectangle_2& box) :rootNode(new Node(box)){}


bool QuadTree::is_leafNode(const std::shared_ptr<Node>& node)
{
	//std::cout.precision(17);
	return node->child.size() == 0;
}

void QuadTree::spilt(std::shared_ptr<Node>& node)
{
	//std::cout.precision(17);
	//CounterClockWise arrangement of Children startting from left bottom
	Point_2 midPoint((node->minCoor.x() + node->maxCoor.x()) / 2, (node->minCoor.y() + node->maxCoor.y()) / 2);

	Iso_rectangle_2 r1(node->minCoor.x(), node->minCoor.y(), midPoint.x(), midPoint.y());
	Iso_rectangle_2 r2(midPoint.x(), node->minCoor.y(), node->maxCoor.x(), midPoint.y());
	Iso_rectangle_2 r3(midPoint.x(), midPoint.y(), node->maxCoor.x(), node->maxCoor.y());
	Iso_rectangle_2 r4(node->minCoor.x(), midPoint.y(), midPoint.x(), node->maxCoor.y());

	node->child.push_back(std::shared_ptr<Node>(new Node(r1)));
	node->child.push_back(std::shared_ptr<Node>(new Node(r2)));
	node->child.push_back(std::shared_ptr<Node>(new Node(r3)));
	node->child.push_back(std::shared_ptr<Node>(new Node(r4)));

	for (int i = 0; i<node->child.size(); ++i){ node->child.at(i)->parent = node; }
}

void QuadTree::insertPoints(std::shared_ptr<Node>& node)
{
	//std::cout.precision(17);
	if (node->parent != NULL)
	{
		for (int i = 0; i<node->parent->insidePoints.size(); ++i)
		{
			if ((node->rectangle.has_on_boundary(node->parent->insidePoints.at(i))) || (node->rectangle.has_on_bounded_side(node->parent->insidePoints.at(i))))
			{
				node->insidePoints.push_back(node->parent->insidePoints.at(i));
				//////////ensure you delete the points in the nodes after split
			}
		}
	}
	else{ std::cout << "This is a root Node" << std::endl; }
}

int QuadTree::checkNumPoints(const Node& node)
{
	//std::cout.precision(17);
	return node.insidePoints.size();
}

std::shared_ptr<Node> generate_QuadTree(QuadTree& tree, std::shared_ptr<Node>& node)
{
	//std::cout.precision(17);
	std::shared_ptr<Node> temp;
	//std::cout<<node->insidePoints.size()<<std::endl;
	if (node->insidePoints.size() >= MAX_NUM)
	{
		tree.spilt(node);
		for (int i = 0; i<node->child.size(); ++i)
		{
			tree.insertPoints(node->child.at(i));
			temp = node->child.at(i);
			//////////ensure you delete the points in the nodes after split

			while (temp->insidePoints.size() >= MAX_NUM)
			{
				temp = generate_QuadTree(tree, temp);
			}
		}
	}
	return temp;
}

void printNode(std::shared_ptr<Node>& node)
{
	//std::cout.precision(17);
	if (node != nullptr)
	{
		std::cout << node->rectangle.vertex(0) << " ";
		std::cout << node->rectangle.vertex(1) << std::endl;
		std::cout << node->rectangle.vertex(1) << " ";
		std::cout << node->rectangle.vertex(2) << std::endl;
		std::cout << node->rectangle.vertex(2) << " ";
		std::cout << node->rectangle.vertex(3) << std::endl;
		std::cout << node->rectangle.vertex(3) << " ";
		std::cout << node->rectangle.vertex(0) << std::endl;

		//std::cout<<node->child.size()<<"Size////////////////////"<<std::endl;
		//std::cout<<node->insidePoints.size()<<"//////////////////////Inside Points"<<std::endl;
	}
	else return;
}

void printTree(const QuadTree& tree)
{
	//std::cout.precision(17);
	std::queue<std::shared_ptr<Node>> nodeQueue;
	std::shared_ptr<Node> temp;
	if (tree.rootNode == nullptr){ return; }
	else
	{
		nodeQueue.push(tree.rootNode);
		while (!nodeQueue.empty())
		{
			temp = nodeQueue.front();
			printNode(temp);
			nodeQueue.pop();
			for (int i = 0; i<temp->child.size(); ++i)
			{
				nodeQueue.push(temp->child.at(i));
			}
		}
	}
}

//Function to get the minimum and Maximum Coordinates of the Point Set
void MinMaxCoor(std::vector<Point_2> &input, Iso_rectangle_2& box){
	//double maxX=0,minX=DBL_MAX,maxY=0,minY=DBL_MAX;
	double maxX = 0, minX = DBL_MAX, maxY = 0, minY = DBL_MAX;
	//std::cout.precision(17);
	std::vector<Point_2>::iterator it = input.begin();
	Point_2 point;
	//string str;
	//while((std::getline(input,str))){
	while (it != input.end()){
		point = *it;
		//istringstream stream(str);
		// while((stream>>point)){
		if (minX>point.x()){ minX = (point.x()); }
		if (minY>point.y()){ minY = (point.y()); }
		if (maxX<point.x()){ maxX = (point.x()); }
		if (maxY<point.y()){ maxY = (point.y()); }
		++it;
	}
	//}
	//std::cout<<maxX<<" "<<minX<<" "<<maxY<<" "<<minY<<std::endl;
	maxX = maxX + 100; maxY = maxY + 100; minX = minX - 100; minY = minY - 100;
	Iso_rectangle_2 bbox(minX, minY, maxX, maxY);
	box = bbox;
}

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
bool on_Line(Point_2 A, Point_2 B, Point_2 C)
{
	//std::cout.precision(17);
	long double dxc = C.x() - A.x();
	long double dyc = C.y() - A.y();

	long double dx1 = B.x() - A.x();
	long double dy1 = B.y() - A.y();

	double cross = ((dxc * dy1) - (dyc * dx1));

	if (cross == 0)
	{
		return 1;
	}

	if (CGAL::abs(dx1) >= CGAL::abs(dy1))
	{
		if (dx1 > 0)
		{
			return (A.x() <= C.x() && C.x() <= B.x());
		}
		else
		{
			return (B.x() <= C.x() && C.x() <= A.x());
		}
	}
	else
	{
		if (dy1 > 0)
		{
			return (A.y() <= C.y() && C.y() <= B.y());
		}
		else
		{
			return (B.y() <= C.y() && C.y() <= A.y());
		}
	}
}
}
#define _CRT_SECURE_NO_WARNINGS

#include "global_datatype.h"
#include "Topology.h"
#include "Geometry.h"
#include "MeshRefinement.h"
#include "CreateDelaunay.h"

int MAX_NUM = 20;
//End of Typedef's
	
std::vector<Point_2> ThreshPoints, NewThreshPoints;
std::vector<vector <Point_2> > Neighbors;
std::vector<Segment_2> Neighbor_Segments;
std::multimap<Edge, Point_2> listing;
//int points_inserted = 0;


Node::Node():minCoor(0,0),maxCoor(0,0),rectangle(minCoor,maxCoor){}
Node::Node(const Point_2& bottom,const Point_2& top):minCoor(bottom),maxCoor(top),rectangle(bottom,top){}
Node::Node(const Iso_rectangle_2& box):rectangle(box),minCoor(box.min()),maxCoor(box.max()){}



QuadTree::QuadTree(){}
//QuadTree::~QuadTree(){}
QuadTree::QuadTree(const Point_2& bottom,const Point_2& top){rootNode->maxCoor=top;rootNode->minCoor= Point_2(bottom);}
QuadTree::QuadTree(const Iso_rectangle_2& box):rootNode(new Node(box)){}


bool QuadTree::is_leafNode(const std::shared_ptr<Node>& node)
{
	//std::cout.precision(17);
	return node->child.size()==0;
}

void QuadTree::spilt(std::shared_ptr<Node>& node)
{
	//std::cout.precision(17);
	//CounterClockWise arrangement of Children startting from left bottom
	Point_2 midPoint((node->minCoor.x()+node->maxCoor.x())/2,(node->minCoor.y()+node->maxCoor.y())/2);
	
	Iso_rectangle_2 r1(node->minCoor.x(),node->minCoor.y(),midPoint.x(),midPoint.y());
	Iso_rectangle_2 r2(midPoint.x(),node->minCoor.y(),node->maxCoor.x(),midPoint.y());
	Iso_rectangle_2 r3(midPoint.x(),midPoint.y(),node->maxCoor.x(),node->maxCoor.y());
	Iso_rectangle_2 r4(node->minCoor.x(),midPoint.y(),midPoint.x(),node->maxCoor.y());
	
	node->child.push_back(std::shared_ptr<Node>(new Node(r1)));
	node->child.push_back(std::shared_ptr<Node>(new Node(r2)));
	node->child.push_back(std::shared_ptr<Node>(new Node(r3)));
	node->child.push_back(std::shared_ptr<Node>(new Node(r4)));
	
	for(int i =0;i<node->child.size();++i){node->child.at(i)->parent=node;}
}

void QuadTree::insertPoints(std::shared_ptr<Node>& node)
{
	//std::cout.precision(17);
	if(node->parent!=NULL)
	{
		for(int i=0;i<node->parent->insidePoints.size();++i)
		{
			if((node->rectangle.has_on_boundary(node->parent->insidePoints.at(i)))||(node->rectangle.has_on_bounded_side(node->parent->insidePoints.at(i))))
			{
				node->insidePoints.push_back(node->parent->insidePoints.at(i));
				//////////ensure you delete the points in the nodes after split
			}
		}
	}
	else{std::cout<<"This is a root Node"<<std::endl;}
}

int QuadTree::checkNumPoints(const Node& node)
{
	//std::cout.precision(17);
	return node.insidePoints.size();
}

std::shared_ptr<Node> generate_QuadTree(QuadTree& tree,std::shared_ptr<Node>& node)
{
	//std::cout.precision(17);
	std::shared_ptr<Node> temp;
	//std::cout<<node->insidePoints.size()<<std::endl;
	if(node->insidePoints.size()>=MAX_NUM)
	{
		tree.spilt(node);
		for(int i=0;i<node->child.size();++i)
		{
			tree.insertPoints(node->child.at(i));
			temp=node->child.at(i);
			//////////ensure you delete the points in the nodes after split

			while(temp->insidePoints.size()>=MAX_NUM)
			{
				temp= generate_QuadTree(tree,temp);
			}
		}
	}
	return temp;
}
/*
void printNode(std::shared_ptr<Node>& node)
{
	//std::cout.precision(17);
	if(node!=nullptr)
	{
		std::cout<<node->rectangle.vertex(0)<<" ";
		std::cout<< node->rectangle.vertex(1)<<std::endl;
		std::cout<< node->rectangle.vertex(1)<<" ";
		std::cout<< node->rectangle.vertex(2)<<std::endl;
		std::cout<< node->rectangle.vertex(2)<<" ";
		std::cout<< node->rectangle.vertex(3)<<std::endl;
		std::cout<< node->rectangle.vertex(3)<<" ";
		std::cout<< node->rectangle.vertex(0)<<std::endl;

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
	if (tree.rootNode==nullptr){return;}
	else
	{
		nodeQueue.push(tree.rootNode);
		while(!nodeQueue.empty())
		{
			temp=nodeQueue.front();
			printNode(temp);
			nodeQueue.pop();
			for(int i=0;i<temp->child.size();++i)
			{
				nodeQueue.push(temp->child.at(i));
			}
		}
	}
}*/

//Function to get the minimum and Maximum Coordinates of the Point Set
void MinMaxCoor(std::vector<Point_2> &input,Iso_rectangle_2& box){
	//double maxX=0,minX=DBL_MAX,maxY=0,minY=DBL_MAX;
	double maxX=0,minX=DBL_MAX,maxY=0,minY=DBL_MAX;
	//std::cout.precision(17);
	std::vector<Point_2>::iterator it = input.begin();
	 Point_2 point;
	//string str;
	//while((std::getline(input,str))){
	while(it != input.end()){
        point = *it;
		//istringstream stream(str);
       // while((stream>>point)){
			if(minX>point.x()){minX=(point.x());}
			if(minY>point.y()){minY=(point.y());}
			if(maxX<point.x()){maxX=(point.x());}
			if(maxY<point.y()){maxY=(point.y());}
			++it;
        }
    //}
	//std::cout<<maxX<<" "<<minX<<" "<<maxY<<" "<<minY<<std::endl;
	maxX=maxX+100;maxY=maxY+100;minX=minX-100;minY=minY-100;
	Iso_rectangle_2 bbox(minX,minY,maxX,maxY);
	box=bbox;
}

bool on_Line(Point_2 A, Point_2 B, Point_2 C)
{
	//std::cout.precision(17);
	long double dxc = C.x() - A.x();
	long double dyc = C.y() - A.y();

	long double dx1 = B.x() - A.x();
	long double dy1 = B.y() - A.y();

	double cross = ((dxc * dy1) - (dyc * dx1));
	
	if(cross == 0)
	{
		return 1;
	}

	if(CGAL::abs(dx1) >= CGAL::abs(dy1))
	{
		if(dx1 > 0)
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
		if(dy1 > 0)
		{
			return (A.y() <= C.y() && C.y() <= B.y());
		}
		else
		{
			return (B.y() <= C.y() && C.y() <= A.y());
		}
	}

}



//To assign ids to the Delaunay Triangulation
void Assign_ids(Delaunay& dt)
{
	//std::cout.precision(17);
	int id = 0;
	Finite_vertices_iterator_2d vit;
	for (vit = dt.finite_vertices_begin(); vit != dt.finite_vertices_end(); vit++) {
		vit->id = id;
		id++;
	}
}

//Function to Output Delaunay Edges so as to view in Opengl


void markFaces(Delaunay dt)
{
	fh f_handle = dt.incident_faces(dt.infinite_vertex());
	std::stack<fh> fh_stack;
	fh_stack.push(f_handle);
	do
	{
		f_handle = fh_stack.top();	
		fh_stack.pop();
	
		if(f_handle->face_is_inside == true)
		{
	
			f_handle->face_is_inside = false;
			for(int i = 0; i < 3; i++)
			{
				if(f_handle->correct_segments[i]==false)
				{							
					fh_stack.push(f_handle->neighbor(i));
				}
	
			}
		}
	
	}while(!fh_stack.empty());	
}

void outputDelaunayEdges(Delaunay& dt, std::ofstream &output)
{
	//output.precision(17);
	/*Assign_ids(dt);

	output.precision(17);
	Finite_vertices_iterator_2d vit;
	for (vit = dt.finite_vertices_begin(); vit != dt.finite_vertices_end(); vit++) {
	output<<vit->point()<<std::endl;
	}*/

	std::ofstream outputRays;
	outputRays.open("OutputVoronoi.txt", std::ofstream::out | std::ofstream::trunc);

	for (fit f1 = dt.finite_faces_begin(); f1 != dt.finite_faces_end(); f1++)
	{
		for (int i = 0; i < 3; i++){
			Edge e = Edge(f1, i);
			//if(!dt.is_infinite(e))
			//{
			CGAL::Object o = dt.dual(e);

			const Segment_2* s = CGAL::object_cast<Segment_2>(&o);
			const Ray_2* r = CGAL::object_cast<Ray_2>(&o);

			//int num_of_intersections=0;
			//Segment_2* temp = new Segment_2;
			//ThreshPoints.clear(); NewThreshPoints.clear();Neighbors.clear(); Neighbor_Segments.clear();				

			if (r)
			{
				outputRays << *r << std::endl;
			}
			if (s)
			{
				outputRays << *s << std::endl;
			}
			if (f1->face_is_inside == true)
			{
				vh first = (f1->vertex(0));
				//unsigned int first_id = first->id;

				vh second = (f1->vertex(1));
				//unsigned int second_id = second->id;

				vh third = (f1->vertex(2));
				//unsigned int third_id = third->id;

				if (output.is_open())
				{
					//output<<"3 "<<first->id<<" "<<second->id<<" "<<third->id<<std::endl;
					output << first->point() << " "; output << second->point() << std::endl;
					output << second->point() << " "; output << third->point() << std::endl;
					output << third->point() << " "; output << first->point() << std::endl;
				}
			}
		}		
	}
	output.close();
}

//Function to Output Delaunay Edges so as to view in Opengl
void outputInsideDelaunayEdges(Delaunay& dt, std::ofstream &output, const QuadTree& tree)
{
	//markFaces(dt);
	std::ofstream outputRays;
	outputRays.open("OutputFinalVoronoi.txt", std::ofstream::out | std::ofstream::trunc);
	/*fh f_handle = dt.incident_faces(dt.infinite_vertex());
	std::stack<fh> fh_stack;
	fh_stack.push(f_handle);
	do
	{
		f_handle = fh_stack.top();
		fh_stack.pop();

		if (f_handle->face_is_inside == true)
		{

			f_handle->face_is_inside = false;
			for (int i = 0; i < 3; i++)
			{
				if (f_handle->correct_segments[i] == false)
				{
					fh_stack.push(f_handle->neighbor(i));
				}

			}
		}

	} while (!fh_stack.empty());*/
	fit f1 = dt.finite_faces_begin();

	for (; f1 != dt.finite_faces_end(); f1++)
	{
		for (int i = 0; i <3; i++){
			Edge e = Edge(f1, i);
			//if(!dt.is_infinite(e))
			//{
			CGAL::Object o = dt.dual(e);

			const Segment_2* s = CGAL::object_cast<Segment_2>(&o);
			const Ray_2* r = CGAL::object_cast<Ray_2>(&o);

			//int num_of_intersections=0;
			//Segment_2* temp = new Segment_2;
			//ThreshPoints.clear(); NewThreshPoints.clear();Neighbors.clear(); Neighbor_Segments.clear();				

			if (r)
			{
				outputRays << *r << std::endl;
			}
			if (s)
			{
				outputRays << *s << std::endl;
			}
			//}
		}
		if (f1->face_is_inside == true)
		{
			//++numInsideTri;
			vh first = (f1->vertex(0));
			vh second = (f1->vertex(1));
			vh third = (f1->vertex(2));

			if (output.is_open())
			{
				output << first->point() << " "; output << second->point() << std::endl;
				output << second->point() << " "; output << third->point() << std::endl;
				output << third->point() << " "; output << first->point() << std::endl;
			}

		}
	}
	output.close();
}
struct EdgeInfo
{
	fh face;
	int vertex_index;
	int index;
};
//Function to Create Voronoi Edges and store it as Segments and Rays in a vector
void create_Voronoi(Delaunay& dt, std::vector<Ray_2>& ray, std::vector<Segment_2>& seg, std::vector<EdgeInfo>& ray_edges, std::vector<EdgeInfo>& seg_edges)
{
	Edge_iterator eit = dt.edges_begin();

	for (; eit != dt.edges_end(); ++eit)
	{
		CGAL::Object o = dt.dual(eit);
		const Segment_2* s = CGAL::object_cast<Segment_2>(&o);
		const Ray_2* r = CGAL::object_cast<Ray_2>(&o);
		if (r)
		{
			ray.push_back(*r);

			EdgeInfo info;
			info.face = eit->first;
			info.vertex_index = eit->second;
			info.index = ray.size() - 1;
			ray_edges.push_back(info);
		}
		else if (s)
		{
			seg.push_back(*s);

			EdgeInfo info;
			info.face = eit->first;
			info.vertex_index = eit->second;
			info.index = seg.size() - 1;
			seg_edges.push_back(info);
		}
	}
}
/*void outputVoro(Delaunay& dt, std::ofstream &outputVor)
{
	std::ofstream outputRays;
	outputVor.open("OutputVor.txt", std::ofstream::out | std::ofstream::trunc);

	Edge_iterator e = dt.edges_begin();

	for(; e != dt.finite_edges_end(); e++)
	//for (fit f1 = dt.finite_faces_begin(); f1 != dt.finite_faces_end(); f1++)
	{
		CGAL::Object o = dt.dual(e);
		const Segment_2* s = CGAL::object_cast<Segment_2>(&o);
		const Ray_2* r = CGAL::object_cast<Ray_2>(&o);
		
		if (outputVor.is_open())
		{
			outputVor << e->first->vertex((e->second + 1) % 3)->point() << " "; outputVor << e->first->vertex((e->second + 2) % 3)->point() << std::endl;
		}
		
	}
	
	outputVor.close();
}*/

int main()
{
	int check;
	std::clock_t t1,t2, t3, t4, t5;
	
	//std::cout.precision(17);
	std::vector<Point_2> OriginalSample,RandomSample;
	
	//Inputs and Outputs from/to various files
	std::ifstream input("neha1.txt");
	int num_of_points=0;
	std::string data;
	while(getline(input,data))
	{
		Point_2 original;
		std::istringstream stream(data);
		while(stream>>original)
		{	
			OriginalSample.push_back(original);
			++num_of_points;
		}
	}
	t1 = clock();
	Iso_rectangle_2	boundingBox;
	MinMaxCoor(OriginalSample,boundingBox);
	QuadTree tree(boundingBox);
	
	tree.rootNode->insidePoints=OriginalSample;
	generate_QuadTree(tree,tree.rootNode);	
	
	//Taking some random sample from the original sample
	t2=clock();
	std::cout << "Total Time Taken for Quadtree: " << ((float)(t2)-(float)(t1)) / CLOCKS_PER_SEC << std::endl;
	/*for(int i=0;i<7;i++)
	{
		int n=std::rand()%(num_of_points-1);
		RandomSample.push_back(OriginalSample.at(n));
		//if(outputRandomSample.is_open()){outputRandomSample<<OriginalSample.at(n)<<std::endl;}
	}*/
	
	std::ifstream inRandomSample;
	inRandomSample.open("SpecialCase_MultiComp1.txt");
	std::string data1;
	while(getline(inRandomSample,data1))
	{
		Point_2 sample;
		std::istringstream stream(data1);
		while(stream>>sample)
		{
			RandomSample.push_back(sample);
			//if(outputRandomSample.is_open()){outputRandomSample<<sample<<std::endl;}			
		}
	}

	//Creating Delaunay Triangulation of the points in the inputRandomSample
	Delaunay dt_sample;
	create_Delaunay(dt_sample, RandomSample);
	//void create_Voronoi(Delaunay& dt, std::vector<Ray_2>& ray, std::vector<Segment_2>& seg, std::vector<EdgeInfo>& ray_edges, std::vector<EdgeInfo>& seg_edges);
	
	//bool iterate=true;
	
	//while(iterate)
	//{  //1
		
		
		std::cout<<".....................................Topology............................."<<std::endl;
		Topology(dt_sample, tree);						
		
		//if(iterate == false)
		//{//4
			t3 = clock();
			std::cout << "Total Time Taken for Topology: " << ((float)(t3)-(float)(t2)) / CLOCKS_PER_SEC << std::endl;			
			
			std::ofstream outputDelaunayTP;
			outputDelaunayTP.open("DelaunayTopologyEdges.txt", std::ofstream::out | std::ofstream::trunc);
			outputDelaunayEdges(dt_sample, outputDelaunayTP);
			//std::cout << "------------------------------------- Save Voronoi edge after Topology --------------------------" << std::endl;
			//std::cin >> check;
			
			std::cout<<"......................Geometric...................."<<std::endl;	
			Geometry(dt_sample, tree);
		
			t4 = clock();
			std::cout << "Total Time Taken for Geometry: " << ((float)(t4)-(float)(t3)) / CLOCKS_PER_SEC << std::endl;
			
			/*std::ofstream outputDelaunayGeo;
			outputDelaunayGeo.open("DelaunayGeometricEdges.txt", std::ofstream::out | std::ofstream::trunc);
			outputDelaunayEdges(dt_sample, outputDelaunayGeo);*/
			//std::ofstream outputDelaunayInsideGeo;
			//outputDelaunayInsideGeo.open("InsideDelaunayGeometricEdges.txt", std::ofstream::out | std::ofstream::trunc);
			//outputInsideDelaunayEdges(dt_sample, outputDelaunayInsideGeo, tree);
			//std::cout << "------------------------------------- Save Voronoi edge after Geometric Approximation --------------------------" << std::endl;
			//std::cin >> check;

			std::cout<<"..............................................Mesh Refinement.................................... "<<std::endl;
			MeshRefinement(dt_sample, tree);
			t5 = clock();
			std::cout << "Total Time Taken for Mesh Refinement: " << ((float)(t5)-(float)(t4)) / CLOCKS_PER_SEC << std::endl;
		//}
			//std::ofstream outputDelaunayMesh;
			//outputDelaunayMesh.open("DelaunayMeshEdges.txt", std::ofstream::out | std::ofstream::trunc);
			//outputDelaunayEdges(dt_sample, outputDelaunayMesh);
			//std::cout << "------------------------------------- Save Voronoi edge after Mesh Refinement --------------------------" << std::endl;
			//std::cin >> check;

			
		
	//}  //1
	
	//float timeSec = ((float)(t5)-(float)(t1)) / CLOCKS_PER_SEC;
	//std::cout << "Total Time Taken for Execution: " << timeSec << std::endl;
	/*std::ofstream outputDelaunayInside;
	outputDelaunayInside.open("InsideDelaunayEdges.txt", std::ofstream::out | std::ofstream::trunc);
	outputInsideDelaunayEdges(dt_sample, outputDelaunayInside, tree);
	
	std::cout<<"Total Points = "<< dt_sample.number_of_vertices()<<std::endl;
	std::cout<<"Total Faces = "<< dt_sample.number_of_faces()<<std::endl;
	
	Assign_ids(dt_sample);
	std::cout << "ply	" << std::endl;

	//std::cout << "ply // format ascii 1.0           { ascii / binary, format version number } //	element vertex 216           { define vertex element, 8 of them in file } //	property float x           { vertex contains float x coordinate } //	property float y           { y coordinate is also a vertex property } //	property float z           { z coordinate is also a vertex property } //	element face 298            { there are 6 face elements in the file } //	property list uchar int vertex_index { vertex_indices is a list of ints } //	end_header " << std::endl;

	for (Vertex_iterator vit = dt_sample.finite_vertices_begin(); vit != dt_sample.finite_vertices_end(); vit++)
	{
		std::cout << vit->point()<<" "<<"0" << std::endl;
	}
	for (fit f = dt_sample.finite_faces_begin(); f != dt_sample.finite_faces_end(); f++)
	{
		if (f->face_is_inside == true)
		{
			//++numInsideTri;
			vh first = (f->vertex(0));
			vh second = (f->vertex(1));
			vh third = (f->vertex(2));
			
			std::cout <<"3"<<" "<< first->id << " " << second->id << " " << third->id << std::endl;			

		}
	}*/
	//std::cout<<"total_circumcenter_position = "<<total_circumcenter_position<<std::endl;
	//std::cout<<"total_inTopology = "<<total_inTopology<<std::endl;
	//std::cout<<"total_inManifold = "<<total_inManifold<<std::endl;
	//std::cout<<"total_inGeometry = "<<total_inGeometry<<std::endl;
	//std::cout<<"total_inEdgeMarking = "<<total_inEdgeMarking<<std::endl;
	
	
	
	int k;
	std::cin>>k;
	return 0;
}
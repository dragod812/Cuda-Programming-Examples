
#include "Topology.h"
#include "Geometry.h"
#include "MeshRefinement.h"


void markEdge_E(Delaunay& dt, const vh& point, const vh& v1, const vh& v2)
{
	//std::ofstream HomoEdges;
	//HomoEdges.open("HomomorphicalEdges.txt",std::ofstream::out | std::ofstream::trunc);

	fh f_handle;
	int i = -1;
	//if (dt.is_edge(v1, v2) || dt.is_edge(v2, v1))
	//{ std::cout << "Edge exists" << std::endl; }
	if (dt.is_edge(point, v1, f_handle, i) || dt.is_edge(v1, point, f_handle, i))
	{
		assert(i >= 0 && i < 3);
		f_handle->correct_segments[i] = true;
		//std::cout << "New Edge	" << point->point() << " " << v1->point() << std::endl;
	}
	if (dt.is_edge(point, v2, f_handle, i) || dt.is_edge(v2, point, f_handle, i))
	{
		assert(i >= 0 && i < 3);
		f_handle->correct_segments[i] = true;
		//std::cout << "New Edge	" << point->point() << " " << v2->point() << std::endl;
	}
	
}

void markEdge_C(Delaunay& dt, const vh& point)
{
	Face_circulator fc = dt.incident_faces(point), d(fc);
	//int n = 0, m = 0;
	do
	{		
		for (int j = 0; j < 3; j++)
		{
			Edge e = Edge(fc, j);
			e.first->correct_segments[e.second] = false;

			if (fc->vertex(j) == point)
			{
				if ((dt.mirror_edge(e)).first->correct_segments[(dt.mirror_edge(e)).second] == true)
				{				
					e.first->correct_segments[e.second] = true;
				}
			}
		}
	} while (++fc != d);
}

void outputInsideDelaunayEdges(Delaunay& dt, std::ofstream &output)
{
	//markFaces(dt);
	//std::ofstream outputRays;
	//outputRays.open("OutputFinalVoronoi.txt", std::ofstream::out | std::ofstream::trunc);
	
	fit f1 = dt.finite_faces_begin();

	for (; f1 != dt.finite_faces_end(); f1++)
	{		
		if (f1->face_is_inside == true && f1->component == 1)
		{
			//++numInsideTri;
			vh first = (f1->vertex(0));
			vh second = (f1->vertex(1));
			vh third = (f1->vertex(2));

			if (output.is_open())
			{
				output << first->point() << " " << second->point()<<std::endl;
				output << second->point() << " " << third->point() << std::endl;
				output << third->point() << " " << first->point() << std::endl;
			}

		}
	}
	output.close();
}

void MeshRefinement(Delaunay& dt_sample, const QuadTree& tree)
{
	/*int comp = -1, cp = 1, comp_cp;
	std::pair<fh, int> face_info;
	//fh f_handle = dt_sample.incident_faces(dt_sample.infinite_vertex());
	Face_circulator fc = dt_sample.incident_faces(dt_sample.infinite_vertex()), done(fc);
	std::stack<pair<fh, int>> fh_stack, comp_stack;

	fh comp_fh, f_handle;

	do {
	fh_stack.push(make_pair(fc, 0)); // not all triangle's comp is 0
	while (!fh_stack.empty())
	{
	f_handle = fh_stack.top().first;
	comp = fh_stack.top().second;

	fh_stack.pop();

	if (f_handle->face_is_inside == false && comp == 0)
	{
	//std::cout << f_handle->vertex(0)->point() << " " << f_handle->vertex(1)->point() << " " << f_handle->vertex(2)->point() << "	Comp =  " << fh_stack.top().second << std::endl;
	//std::cout << "1" << std::endl;
	for (int i = 0; i < 3; i++)
	{
	if (f_handle->correct_segments[i] == true)
	{
	//std::cout << "2" << std::endl;
	//fh fhand = f_handle->neighbor(i);
	comp_stack.push(make_pair(f_handle->neighbor(i), cp)); // check neighbor, it might be outside triangle
	//std::cout << fhand->vertex(0)->point() << " " << fhand->vertex(1)->point() << " " << fhand->vertex(2)->point() << "	Comp =  " << fh_stack.top().second << std::endl;
	if (comp_stack-)
	while (!comp_stack.empty())
	{
	//std::cout << "3" << std::endl;
	std::cout << comp_stack.top().first->vertex(0)->point() << " " << comp_stack.top().first->vertex(1)->point() << " " << comp_stack.top().first->vertex(2)->point() <<  std::endl;
	comp_fh = comp_stack.top().first;
	comp_cp = comp_stack.top().second;
	comp_fh->face_is_inside = true;
	comp_stack.pop();

	for (int i = 0; i < 3; i++)
	{
	if (comp_fh->correct_segments[i] == false && comp_fh->neighbor(i)->face_is_inside == false)  // ( && comp_cp != cp) added second condition
	{
	comp_stack.push(make_pair(comp_fh->neighbor(i), cp));   // here going into infinite loop
	}
	}
	}
	}
	}
	}

	}
	//++cp;
	} while (++fc != done);
	*/
	int cp = 1;
	//int comp = -1, cp = 1, comp_cp;
	//std::pair<fh, int> face_info;
	//fh f_handle = dt_sample.incident_faces(dt_sample.infinite_vertex());
	Face_circulator fc = dt_sample.incident_faces(dt_sample.infinite_vertex()), done(fc);
	std::stack<fh> fh_stack, comp_stack;

	fh comp_fh, f_handle;

	do {
		fh_stack.push(fc); // not all triangle's comp is 0
		while (!fh_stack.empty())
		{
			f_handle = fh_stack.top();

			fh_stack.pop();

			if (f_handle->face_is_inside == false && f_handle->component == 0)
			{
				//std::cout << f_handle->vertex(0)->point() << " " << f_handle->vertex(1)->point() << " " << f_handle->vertex(2)->point() << "	Comp =  " << fh_stack.top().second << std::endl;
				//std::cout << "1" << std::endl;
				for (int i = 0; i < 3; i++)
				{
					if (f_handle->correct_segments[i] == true && f_handle->neighbor(i)->component != 1)
					{
						comp_stack.push(f_handle->neighbor(i)); // check neighbor, it might be outside triangle
						//std::cout << fhand->vertex(0)->point() << " " << fhand->vertex(1)->point() << " " << fhand->vertex(2)->point() << "	Comp =  " << fh_stack.top().second << std::endl;

						while (!comp_stack.empty())
						{
							//std::cout << "3" << std::endl;
							std::cout << comp_stack.top()->vertex(0)->point() << " " << comp_stack.top()->vertex(1)->point() << " " << comp_stack.top()->vertex(2)->point() << "	cp	=	" << cp << std::endl;
							comp_fh = comp_stack.top();
							comp_fh->component = cp;
							comp_fh->face_is_inside = true;
							comp_stack.pop();

							for (int i = 0; i < 3; i++)
							{
								if (comp_fh->correct_segments[i] == false && comp_fh->neighbor(i)->face_is_inside == false)  // ( && comp_cp != cp) added second condition
								{
									//comp_fh->neighbor(i)->component = 1;
									comp_stack.push(comp_fh->neighbor(i));   // here going into infinite loop
								}
							}
						} // comp_stack end of while
						//return;
						++cp;
					}
				}

			}

		}
		//++cp;
	} while (++fc != done);


	Finite_faces_iterator_2d face_iterator;
	bool flag1 = true;
	bool flag; Point_2 P;
	//int n=0;
	int c = 0, d = 0, k = 1;

	while (flag1)
	{
		flag1 = false;
		vh v1, v2;
		for (face_iterator = dt_sample.finite_faces_begin(); face_iterator != dt_sample.finite_faces_end(); ++face_iterator)
		{
			if (face_iterator->face_is_inside == true && face_iterator->component == 1)
			{
				//if (CGAL::area(face_iterator->vertex(0)->point(), face_iterator->vertex(1)->point(), face_iterator->vertex(2)->point()) >= 0.000001)
				{
					//std::cout << "1" << std::endl;
					bool skinny_triangle = false;
					bool Circumcenter_is_outside = false;

					Point_2 circum_center, enchroch_Point;
					skinny_triangle = isSkinnyTriangle(face_iterator);
					
					if (skinny_triangle)
					{
						//std::cout << "skinny triangle	" <<  std::endl; //++c;
						circumcenter_position(dt_sample, face_iterator, Circumcenter_is_outside, circum_center, enchroch_Point, v1, v2, flag);
						//std::cout << "circumcenter position found" << std::endl;
						if (flag == true)
						{
							//std::cout<<"Enchrochment"<<std::endl;
							P = enchroch_Point;
							flag1 = true;
							break;
						}
						else if (Circumcenter_is_outside == false)
						{
							P = circum_center;
							//std::cout<<"Circumcenter"<<std::endl;	
							flag1 = true;
							break;
						}
					}
				}
			}
		}
		if (flag1 == true)
		{
			if (flag == true)
			{
				//std::cout << "1" << std::endl;
				vh tempVhandle = dt_sample.insert(P);

				markEdge_E(dt_sample, tempVhandle, v1, v2);
				mark_InsideFace(dt_sample, tempVhandle, k);
				std::cout << tempVhandle->point() << std::endl;
				//std::ofstream outputDelaunayInside;
				//outputDelaunayInside.open("InsideDelaunayEdges.txt", std::ofstream::out | std::ofstream::trunc);
				//outputInsideDelaunayEdges(dt, outputDelaunayInside);
				//std::cin >> check;
			}
			if (flag == false)
			{
				//std::cout << "2" << std::endl;
				vh tempVhandle = dt_sample.insert(P);

				markEdge_C(dt_sample, tempVhandle);
				mark_InsideFace(dt_sample, tempVhandle, k);
				std::cout << tempVhandle->point() << std::endl;
				//std::ofstream outputDelaunayInside;
				//outputDelaunayInside.open("InsideDelaunayEdges.txt", std::ofstream::out | std::ofstream::trunc);
				//outputInsideDelaunayEdges(dt, outputDelaunayInside);

			}
			//Topology(dt_sample, tree);
			//Geometry(dt_sample, tree);
			if (d >= 8)
			{
				std::ofstream outputDelaunayInside;
				outputDelaunayInside.open("InsideDelaunayEdges.txt", std::ofstream::out | std::ofstream::trunc);
				outputInsideDelaunayEdges(dt_sample, outputDelaunayInside);
			}
			++d;
		}
		//std::cout << "In while loop " << std::endl;		

	}
	/*++k;
	flag1 = true;
	while (flag1)
	{
		flag1 = false;
		vh v1, v2;
		for (face_iterator = dt_sample.finite_faces_begin(); face_iterator != dt_sample.finite_faces_end(); ++face_iterator)
		{
			if (face_iterator->face_is_inside == true && face_iterator->component == k)
			{
				if (CGAL::area(face_iterator->vertex(0)->point(), face_iterator->vertex(1)->point(), face_iterator->vertex(2)->point()) >= 0.000001)
				{
					//std::cout << "1" << std::endl;
					bool skinny_triangle = false;
					bool Circumcenter_is_outside = false;

					Point_2 circum_center, enchroch_Point;
					skinny_triangle = isSkinnyTriangle(face_iterator);
					if (skinny_triangle)
					{
						//std::cout << "skinny triangle	" <<  std::endl; ++c;
						circumcenter_position(dt_sample, face_iterator, Circumcenter_is_outside, circum_center, enchroch_Point, v1, v2, flag);
						//std::cout << "3" << std::endl;
						if (flag == true)
						{
							//std::cout<<"Enchrochment"<<std::endl;
							P = enchroch_Point;
							flag1 = true;
							break;
						}
						else if (Circumcenter_is_outside == false)
						{
							P = circum_center;
							//std::cout<<"Circumcenter"<<std::endl;	
							flag1 = true;
							break;
						}
					}
				}
			}
		}
		if (flag1 == true)
		{
			if (flag == true)
			{
				//std::cout << "enchrochment" << std::endl;
				vh tempVhandle = dt_sample.insert(P);

				markEdge_E(dt_sample, tempVhandle, v1, v2);
				mark_InsideFace(dt_sample, tempVhandle);
				//std::cout << tempVhandle->point() << std::endl;
				//std::ofstream outputDelaunayInside;
				//outputDelaunayInside.open("InsideDelaunayEdges.txt", std::ofstream::out | std::ofstream::trunc);
				//outputInsideDelaunayEdges(dt, outputDelaunayInside);
				//std::cin >> check;
			}
			if (flag == false)
			{
				//std::cout << "circumcenter inside" << std::endl;

				vh tempVhandle = dt_sample.insert(P);

				markEdge_C(dt_sample, tempVhandle);
				mark_InsideFace(dt_sample, tempVhandle);
				//std::cout << tempVhandle->point() << std::endl;
				//std::ofstream outputDelaunayInside;
				//outputDelaunayInside.open("InsideDelaunayEdges.txt", std::ofstream::out | std::ofstream::trunc);
				//outputInsideDelaunayEdges(dt, outputDelaunayInside);

			}
			//Topology(dt_sample, tree);
			//Geometry(dt_sample, tree);
		}
		//std::cout << "In while loop " << std::endl;

	}*/


	/*if (c == 0){
		//try{
		//for (Edge_iterator ez = dt_sample.edges_begin(); ez != dt_sample.edges_end(); ++ez)
		for (face_iterator = dt_sample.finite_faces_begin(); face_iterator != dt_sample.finite_faces_end(); ++face_iterator)
		{
		for (int m = 0; m < 3; ++m){
		//std::cout << "Edge	" << ez->first->vertex((ez->second + 1) % 3)->point() << " " << ez->first->vertex((ez->second + 2) % 3)->point() << std::endl;
		if (face_iterator->correct_segments[m] == true){
		Edge fe = Edge(face_iterator, m);
		//if (ez->first->correct_segments[ez->second] == true){
		// && (CGAL::squared_distance(ez->first->vertex((ez->second + 1) % 3)->point(), ez->first->vertex((ez->second + 2) % 3)->point()) > 4)
		//std::cout << "Restricted Edge	" << ez->first->vertex((ez->second + 1) % 3)->point() << " " << ez->first->vertex((ez->second + 2) % 3)->point() << std::endl;
		//Point_2 mP = CGAL::midpoint(ez->first->vertex((ez->second + 1) % 3)->point(), ez->first->vertex((ez->second + 2) % 3)->point());//et->first->vertex((et->second + 1) % 3)->point(), et->first->vertex((et->second + 2) % 3)->point());
		Point_2 mP = CGAL::midpoint(fe.first->vertex((fe.second + 1) % 3)->point(), fe.first->vertex((fe.second + 2) % 3)->point());
		std::cout << "midPoint	= " << mP << std::endl;
		//std::cout << "2" << std::endl;

		vh tempVhandle = dt_sample.insert(mP);
		//std::cout << "condition true" << std::endl;
		vh v1 = fe.first->vertex((fe.second + 1) % 3);
		vh v2 = fe.first->vertex((fe.second + 2) % 3);

		markEdge_E(dt_sample, tempVhandle, v1, v2);
		mark_InsideFace(dt_sample, tempVhandle);

		}
		}
		}
		std::cout << "done" << std::endl;
		//}
		//catch (const std::exception &exc){
		//	std::cout << "exception" << exc.what() << std::endl;
		//}
		++c;
		//flag1 = true;
		//break;
		//std::cout << "done Now" << std::endl;
		}*/
	//++d;
	//std::cout << "d	=	" << d << std::endl;
	//}
}


#include "Skinny.h"
void mark_InsideFace(Delaunay& dt, const vh& point, int& k)
{
	Face_circulator fc = dt.incident_faces(point), done(fc);

	/*do
	{
		if (fc->face_is_inside == true)
		{
			//fc->face_is_inside = true;
			for (int i = 0; i < 3; i++)
			{
				if (fc->correct_segments[i] == false)
				if ((fc->neighbor(i))->face_is_inside == false)
					fc->face_is_inside = false;
			}
		}
		//fc->face_is_inside = true;
		if (fc->face_is_inside == true)
			fc->component = k;

	} while (++fc != done);*/
	do
	{
		if (fc->face_is_inside == false)
		{
			//fc->face_is_inside = true;
			for (int i = 0; i < 3; i++)
			{
				if (fc->correct_segments[i] == false)
				if ((fc->neighbor(i))->face_is_inside == true)
					fc->face_is_inside = true;
			}
		}
		//fc->face_is_inside = true;
		if (fc->face_is_inside == true)
			fc->component = k;

	} while (++fc != done);
}

bool isSkinnyTriangle(fh faceItr)
{
	//std::cout.precision(17);
	double circumradius = 0;
	double s_distance = 0;
	bool answer = false;

	vh first = (faceItr->vertex(0));
	unsigned int first_id = first->id;

	vh second = (faceItr->vertex(1));
	unsigned int second_id = second->id;

	vh third = (faceItr->vertex(2));
	unsigned int third_id = third->id;

	Point onePt = first->point();
	Point twoPt = second->point();
	Point threePt = third->point();

	//Find Circumradius
	circumradius = (CGAL::squared_radius(onePt, twoPt, threePt));
	//std::cout<<"circumradius is		"<<circumradius<<std::endl;						
	double dist1 = (CGAL::squared_distance(onePt, twoPt));
	double dist2 = (CGAL::squared_distance(twoPt, threePt));
	double dist3 = (CGAL::squared_distance(threePt, onePt));

	//c is the circumcenter of the concerned triangle
	Point c = CGAL::circumcenter(onePt, twoPt, threePt);
	//std::cout<<"Circumcenter is		"<<c<<std::endl;
	//s_distance is the smallest edge length
	if (dist1 < dist2 && dist1 < dist3)
		s_distance = dist1;
	else if (dist2 < dist1 && dist2 < dist3)
		s_distance = dist2;
	else
		s_distance = dist3;


	if (circumradius / s_distance >= 2)				// SKINNY TRIANGLES
	{
		//skinny_bound_three++;
		answer = true;
		//std::cout<<"it is a skinny triangle"<<std::endl;
	}
	return answer;
}

bool is_circumcenter_inside(Point_2 p1, Point_2 p2, Point_2 p3, Point_2 q)
{
	if (CGAL::collinear(p1, p2, q) || CGAL::collinear(p2, p3, q) || CGAL::collinear(p3, p1, q))
	{
		return 1;
	}

	// if orientation(p1,p2,q) == orientation(p2,p3,q) == orientation (p3, p1, q)
	//then retun 1 else return 0

	else if ((CGAL::orientation(p1, p2, q) == CGAL::orientation(p2, p3, q)) && (CGAL::orientation(p2, p3, q) == CGAL::orientation(p3, p1, q)))
	{
		return 1;
	}
	else
	{
		return 0;
	}
}

void circumcenter_position(Delaunay& dt, fh faceItrs, bool &circumcenter_is_outside, Point_2& circum_center, Point_2& enchroch_Point, vh& v1, vh& v2, bool& flag)
{
	int not_to_check = -1;
	circumcenter_is_outside = true;
	fh next_face;
	bool move_ahead = true;

	vh first = (faceItrs->vertex(0));
	vh second = (faceItrs->vertex(1));
	vh third = (faceItrs->vertex(2));

	//Find centroid
	Point_2 centroid = CGAL::centroid(first->point(), second->point(), third->point());
	Point_2 c = CGAL::circumcenter(first->point(), second->point(), third->point());
	std::cout << "Starting Face	=	" << first->point() << " " << second->point() << " " << third->point()<< std::endl;
	std::cout << "centroid	=	" << centroid << "	circumcenter	=	" << c << std::endl;
	flag = false;
	/*for (Edge_iterator et = dt.finite_edges_begin(); et != dt.finite_edges_end(); ++et)
	{
	if (et->first->correct_segments[et->second] == true)
	{
	Point_2 fp1 = et->first->vertex(0)->point();
	Point_2 fp2 = et->first->vertex(1)->point();
	Point_2 fp3 = et->first->vertex(2)->point();

	Point_2 cc = CGAL::circumcenter(fp1, fp2, fp3);
	//long double rad;
	//std::cout << "1" << std::endl;
	//if (!(CGAL::collinear(fp1, fp2, fp3))){
	//rad = CGAL::squared_radius(fp1, fp2, fp3);
	//std::cout << "2" << std::endl;
	if ((CGAL::squared_distance(cc, c)) <= (CGAL::squared_radius(fp1, fp2, fp3))){
	std::cout << "1" << std::endl;
	Point_2 midPoint = CGAL::midpoint(et->first->vertex((et->second + 1) % 3)->point(), et->first->vertex((et->second + 2) % 3)->point());
	if (CGAL::squared_distance(midPoint, et->first->vertex((et->second + 1) % 3)->point()) >= CGAL::squared_distance(midPoint, c))
	{
	std::cout << "2" << std::endl;
	flag = true;
	enchroch_Point = midPoint;
	v1 = et->first->vertex((et->second + 1) % 3);
	v2 = et->first->vertex((et->second + 2) % 3);
	break;
	}
	//}
	}
	}
	}*/
	if (flag == false)
	{
		Point_2 p1(centroid.x(), centroid.y());
		Point_2 p2(c.x(), c.y());

		// Line between centroid and circumcenter.
		Segment_2 l(p1, p2);
		Point_2 p_first, p_second, p_third;
		Segment_2 line_1, line_2, line_3;
		//++total_circumcenter_position;

		while (move_ahead)
		{

			//std::cout<<" Inside loop "<<std::endl;					
			p_first = Point_2(first->point().x(), first->point().y());
			p_second = Point_2(second->point().x(), second->point().y());
			p_third = Point_2(third->point().x(), third->point().y());

			line_1 = Segment_2(p_first, p_second);  // Line between first and second point of triangle
			line_2 = Segment_2(p_second, p_third);	// Line between second and third point of triangle
			line_3 = Segment_2(p_third, p_first);	// Line between third and first point of triangle
			//line_1 = Segment_2(Edge(faceItrs, 0));
			//Edge e2 = Edge(faceItrs, 1);
			//Edge e3 = Edge(faceItrs, 2);

			// FIRST CHECK IF CIRCUMCENTER IS INSIDE THIS TRIANGLE
			if (is_circumcenter_inside(p_first, p_second, p_third, p2)) //If circumcenter is inside the same starting face
			{
				std::cout<<" circumcenter_inside is true "<<std::endl;
				if (!faceItrs->face_is_inside)		//face is outside
				{
					circumcenter_is_outside = true;
				}
				else
				{
					circumcenter_is_outside = false;
				}
				move_ahead = false;
				break;
			}
			else if ((!CGAL::do_intersect(l, line_1)) && (!CGAL::do_intersect(l, line_2)) && (!CGAL::do_intersect(l, line_3)))
			{
				std::cout << "No Intersection in circumcenter position" << std::endl;
				circumcenter_is_outside = false;
				move_ahead = false;
				break;
			}
			else
			{
				/*if ((faceItrs->correct_segments[0] == true) && (!CGAL::do_intersect(l, line_2)))
				{
					if (CGAL::do_intersect(l, line_2) == true)
					{
						move_ahead = false;
						break;
					}
					if (CGAL::do_intersect(l, line_1) == true)
						next_face = faceItrs->neighbor(2);
					else if (CGAL::do_intersect(l, line_3) == true)
						next_face = faceItrs->neighbor(1);
				}
				if ((faceItrs->correct_segments[1] == true) && (!CGAL::do_intersect(l, line_3)))
				{
					if (CGAL::do_intersect(l, line_3) == true)
					{
						move_ahead = false;
						break;
					}
					if (CGAL::do_intersect(l, line_1) == true)
						next_face = faceItrs->neighbor(2);
					else if (CGAL::do_intersect(l, line_2) == true)
						next_face = faceItrs->neighbor(0);
				}
				if ((faceItrs->correct_segments[2] == true) && (!CGAL::do_intersect(l, line_1)))
				{
					if (CGAL::do_intersect(l, line_1) == true)
					{
						move_ahead = false;
						break;
					}
					if (CGAL::do_intersect(l, line_2) == true)
						next_face = faceItrs->neighbor(0);
					else if (CGAL::do_intersect(l, line_3) == true)
						next_face = faceItrs->neighbor(1);
				}
				//not_to_check = next_face->index(faceItrs);
				//assert(next_face->neighbor(not_to_check) == faceItrs);
				//faceItrs = next_face;
			*/
				if ((not_to_check == -1 || not_to_check != 2) && (faceItrs->correct_segments[2] == false))
				{
					if (CGAL::do_intersect(l, line_1) == true)
					{
						next_face = faceItrs->neighbor(2);
						std::cout << line_1.source() << " " << line_1.end() << std::endl;
					}
				}
				if ((not_to_check == -1 || not_to_check != 0) && (faceItrs->correct_segments[0] == false))
				{
					if (CGAL::do_intersect(l, line_2) == true)

					{
						next_face = faceItrs->neighbor(0);
						std::cout << line_2.source() << " " << line_2.end() << std::endl;
					}
				}
				if ((not_to_check == -1 || not_to_check != 1) && (faceItrs->correct_segments[1] == false))
				{
					if (CGAL::do_intersect(l, line_3) == true)
					{
						next_face = faceItrs->neighbor(1);
						std::cout << line_3.source() << " " << line_3.end() << std::endl;
					}
				}
				not_to_check = next_face->index(faceItrs);
				assert(next_face->neighbor(not_to_check) == faceItrs);
				faceItrs = next_face;
				
			}
			first = (faceItrs->vertex(0));
			//first_id = first->id; 

			second = (faceItrs->vertex(1));
			//second_id = second->id;

			third = (faceItrs->vertex(2));
			//third_id = third->id;	

			std::cout << "Nxt face cordinates	" << first->point() << " " << second->point() << " " << third->point() << std::endl;
			/*else
			{
				if ((!CGAL::do_intersect(l, line_1)) && (!CGAL::do_intersect(l, line_2)) && (!CGAL::do_intersect(l, line_3)))
				//if ((!CGAL::do_intersect(l, e1)) && (!CGAL::do_intersect(l, e2)) && (!CGAL::do_intersect(l, e3)))
				{
					std::cout << "No Intersection in circumcenter position" << std::endl;
					circumcenter_is_outside = false;
					move_ahead = false;
					break;
				}
				//std::cout<<"cirumcenter is not inside and not_to_check = "<<not_to_check<<std::endl;
				if (not_to_check == -1 || not_to_check != 2){
					//std::cout<<" 1. "<<std::endl;
					if (CGAL::do_intersect(l, line_1) == true)
					{
						//std::cout<<"  1"<<std::endl;
						//if(not_to_check==-1 || not_to_check == 2)
						//{ 
						next_face = faceItrs->neighbor(2);
						//std::cout<<"its not checking 2"<<std::endl;
					}
				}
				if (not_to_check == -1 || not_to_check != 0){
					//std::cout<<" 2. "<<std::endl;
					if (CGAL::do_intersect(l, line_2) == true)
					{
						//std::cout<<"  2"<<std::endl;
						//if(not_to_check == -1 || not_to_check == 0)
						//{ 
						next_face = faceItrs->neighbor(0);
						//std::cout<<"its not checking 0"<<std::endl;
					}
				}
				if (not_to_check == -1 || not_to_check != 1){
					//std::cout<<" 3. "<<std::endl;
					if (CGAL::do_intersect(l, line_3) == true)
					{
						//std::cout<<"  3"<<std::endl;
						//if(not_to_check == -1 || not_to_check == 1)
						//{
						next_face = faceItrs->neighbor(1);
						//std::cout<<"its not checking 1"<<std::endl;
					}
				}

				not_to_check = next_face->index(faceItrs);
				assert(next_face->neighbor(not_to_check) == faceItrs);
				faceItrs = next_face;
				//std::cout<<"got the nxt face"<<std::endl;
			}
			*/			
		}
		Point_2 cc = CGAL::circumcenter(first->point(), second->point(), third->point());
		if ((CGAL::squared_distance(cc, c)) <= (CGAL::squared_radius(first->point(), second->point(), third->point()))){
			
			for (int k = 0; k < 2; k++)
			{
				if (faceItrs->correct_segments[k] == true && faceItrs->component == 1){
					Edge et = Edge(faceItrs, k);
					Point_2 midPoint = CGAL::midpoint(et.first->vertex((et.second + 1) % 3)->point(), et.first->vertex((et.second + 2) % 3)->point());//et->first->vertex((et->second + 1) % 3)->point(), et->first->vertex((et->second + 2) % 3)->point());
					if (CGAL::squared_distance(midPoint, et.first->vertex((et.second + 1) % 3)->point()) >= CGAL::squared_distance(midPoint, c))
					{
						std::cout << "enchroachment" << std::endl;
						flag = true;
						enchroch_Point = midPoint;
						v1 = et.first->vertex((et.second + 1) % 3);
						v2 = et.first->vertex((et.second + 2) % 3);
						break;
					}
				}
			}
		}

	}
	circum_center = c;
}


#ifndef SKINNY_INCLUDED
#define SKINNY_INCLUDED

#include "global_datatype.h"

void mark_InsideFace(Delaunay& dt, const vh& point, int& k);
bool isSkinnyTriangle(fh faceItr);
bool is_circumcenter_inside(Point_2 p1, Point_2 p2, Point_2 p3, Point_2 q);
void circumcenter_position(Delaunay& dt, fh faceItrs, bool &circumcenter_is_outside, Point_2& circum_center, Point_2& enchroch_Point, vh& v1, vh& v2, bool& flag);
#endif //SKINNY_INCLUDED
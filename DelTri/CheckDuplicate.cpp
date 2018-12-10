
//#include "CheckDuplicate.h"
#include "global_datatype.h"
void check_duplicate(std::vector<Point_2> &input)
{
	//std::cout.precision(17);
	//std::vector<Point_2> th_pts;
	for (int j = 0; j<input.size(); ++j)
	{
		for (int k = j; k<input.size(); ++k)
		{
			if (j == k){ continue; }
			else
			{
				if (input.at(j) == input.at(k))
				{
					//ThreshPoints.erase(ThreshPoints.begin(),ThreshPoints.begin()+k);	--k;
					input.erase(input.begin() + k);	--k;
				}

			}
		}
	}
}
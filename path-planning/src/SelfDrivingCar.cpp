//===--- SelfDrivingCar--.cpp -----------------------------------*- C++ -*-===//
//
// This source file is part of the Bosch Challenge Path Planner Project
//
// Author: Luis Vivero <luis.a.vivero@gmail.com>
//
//===----------------------------------------------------------------------===//
//  Udacity Open Source Self-Driving Car
//  Copyright (C) 2016, 2017 Udacity
//
//  This program is free software; you can redistribute it and/or
//  modify it under the terms of the GNU General Public License
//  as published by the Free Software Foundation; either version 2
//  of the License, or (at your option) any later version.
//
//  This program is distributed in the hope that it will be useful,
//  but WITHOUT ANY WARRANTY; without even the implied warranty of
//  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//  GNU General Public License for more details.
//
//===----------------------------------------------------------------------===//

#include "SelfDrivingCar.h"


using namespace std;

constexpr double pi() { return M_PI; }

// For converting back and forth between radians and degrees.
double deg2rad(double x) { return x * pi() / 180; }
double rad2deg(double x) { return x * 180 / pi(); }

double distance(double x1, double y1, double x2, double y2) {
  return sqrt((x2-x1)*(x2-x1)+(y2-y1)*(y2-y1));
}

int ClosestWaypoint(double x, double y, const vector<double> &maps_x,
                    const vector<double> &maps_y)
{

  double closestLen = 100000; //large number
  int closestWaypoint = 0;

  for(int i = 0; i < maps_x.size(); i++)
  {
    double map_x = maps_x[i];
    double map_y = maps_y[i];
    double dist = distance(x,y,map_x,map_y);
    if(dist < closestLen)
    {
      closestLen = dist;
      closestWaypoint = i;
    }

  }

  return closestWaypoint;

}

int NextWaypoint(double x, double y, double theta,
                 const vector<double> &maps_x, const vector<double> &maps_y)
{

  int closestWaypoint = ClosestWaypoint(x,y,maps_x,maps_y);

  double map_x = maps_x[closestWaypoint];
  double map_y = maps_y[closestWaypoint];

  double heading = atan2( (map_y-y),(map_x-x) );

  double angle = abs(theta-heading);

  if(angle > pi()/4)
  {
    closestWaypoint++;
  }

  return closestWaypoint;

}


// Transform from Frenet s,d coordinates to Cartesian x,y
vector<double> getXY(double s, double d, const vector<double> &maps_s,
                     const vector<double> &maps_x, const vector<double> &maps_y)
{
  int prev_wp = -1;

  while(s > maps_s[prev_wp+1] && (prev_wp < (int)(maps_s.size()-1) ))
  {
    prev_wp++;
  }

  int wp2 = (prev_wp+1)%maps_x.size();

  double heading = atan2((maps_y[wp2]-maps_y[prev_wp]),(maps_x[wp2]-maps_x[prev_wp]));
  // the x,y,s along the segment
  double seg_s = (s-maps_s[prev_wp]);

  double seg_x = maps_x[prev_wp]+seg_s*cos(heading);
  double seg_y = maps_y[prev_wp]+seg_s*sin(heading);

  double perp_heading = heading-pi()/2;

  double x = seg_x + d*cos(perp_heading);
  double y = seg_y + d*sin(perp_heading);

  return {x,y};

}

// Transform from Cartesian x,y coordinates to Frenet s,d coordinates
vector<double> getFrenet(double x, double y, double theta,
                         const vector<double> &maps_x, const vector<double> &maps_y)
{
  int next_wp = NextWaypoint(x,y, theta, maps_x,maps_y);

  int prev_wp;
  prev_wp = next_wp-1;
  if(next_wp == 0)
  {
    prev_wp  = maps_x.size()-1;
  }

  double n_x = maps_x[next_wp]-maps_x[prev_wp];
  double n_y = maps_y[next_wp]-maps_y[prev_wp];
  double x_x = x - maps_x[prev_wp];
  double x_y = y - maps_y[prev_wp];

  // find the projection of x onto n
  double proj_norm = (x_x*n_x+x_y*n_y)/(n_x*n_x+n_y*n_y);
  double proj_x = proj_norm*n_x;
  double proj_y = proj_norm*n_y;

  double frenet_d = distance(x_x,x_y,proj_x,proj_y);

  //see if d value is positive or negative by comparing it to a center point

  double center_x = 1000-maps_x[prev_wp];
  double center_y = 2000-maps_y[prev_wp];
  double centerToPos = distance(center_x,center_y,x_x,x_y);
  double centerToRef = distance(center_x,center_y,proj_x,proj_y);

  if(centerToPos <= centerToRef)
  {
    frenet_d *= -1;
  }

  // calculate s value
  double frenet_s = 0;
  for(int i = 0; i < prev_wp; i++)
  {
    frenet_s += distance(maps_x[i],maps_y[i],maps_x[i+1],maps_y[i+1]);
  }

  frenet_s += distance(0,0,proj_x,proj_y);

  return {frenet_s,frenet_d};

}

Path SelfDrivingCar::Route(const Path &previous_path,
                           const Vehicle &ego_car,
                           const vector<RaceCar> &racers,
                           bool two_lane_change) {

  int frontcar_idx = road.FindFrontCarInLane(ego_car, racers, ego_lane);

  // cout << " frontcar_idx:" << frontcar_idx << endl;

  bool is_free_to_go = true;
  bool match_frontcar_speed = false;
  auto front_car_speed = kRefVel * 2;
  if (frontcar_idx != -1) {
    const RaceCar & car(racers[frontcar_idx]);
    double response_time = kTimeInterval * kPathLength;
    auto safe_dist = ego_car.speed * kMilesPerHourToMetersPerSecond
                     * response_time + kFrontCarSafeCarDistance;
    auto dist = car.s - ego_car.s;
    front_car_speed = sqrt(car.vx * car.vx + car.vy * car.vy);
    cout << " frontcar dist:" << dist << endl;
    cout << " safe dist:" << safe_dist << endl;
    //is_free_to_go = (dist >= safe_dist);
    if (dist <= safe_dist) {
    //if (dist <= kFrontCarSafeCarDistance) {
      is_free_to_go = false;
      if (ego_car.speed > front_car_speed) {
        match_frontcar_speed = true;
      }
    }
  }

  //cout << boolalpha;
  cout << "is_free_to_go:" << is_free_to_go << endl;
  cout << "front_car_speed:" << front_car_speed << endl;

  if (!is_free_to_go) {
    if (match_frontcar_speed) {
      if (refVelocity > (front_car_speed * 0.9)) {
        cout << "match_frontcar_speed..." << endl;
        refVelocity -= 0.442; //(refVelocity * kTimeInterval);
      } else {
        refVelocity = front_car_speed;
      }
    }
  } else {
    if (refVelocity < kRefVel) {
      cout << "accelerate" << is_free_to_go << endl;
      refVelocity += 0.442; //(refVelocity * kTimeInterval);
    }
  }

  vector<double> ptsx;
  vector<double> ptsy;

  double ref_x = ego_car.x;
  double ref_y = ego_car.y;
  double ref_yaw = deg2rad(ego_car.yaw);

  int prev_size = previous_path.X.size();
  if ( prev_size < 2 ) {
    double prev_car_x = ego_car.x - cos(ego_car.yaw);
    double prev_car_y = ego_car.y - sin(ego_car.yaw);

    ptsx.push_back(prev_car_x);
    ptsx.push_back(ego_car.x);

    ptsy.push_back(prev_car_y);
    ptsy.push_back(ego_car.y);
  }
  else {

    ref_x = previous_path.X[prev_size - 1];
    ref_y = previous_path.Y[prev_size - 1];

    double ref_x_prev = previous_path.X[prev_size - 2];
    double ref_y_prev = previous_path.Y[prev_size - 2];
    ref_yaw = atan2(ref_y - ref_y_prev, ref_x - ref_x_prev);

    ptsx.push_back(ref_x_prev);
    ptsx.push_back(ref_x);

    ptsy.push_back(ref_y_prev);
    ptsy.push_back(ref_y);
  }

  cout << boolalpha;
  cout << "two_lane_change? " << two_lane_change << endl;

  int wp0_s = two_lane_change ? long_spline[0] : short_spline[0];
  int wp1_s = two_lane_change ? long_spline[1] : short_spline[1];
  int wp2_s = two_lane_change ? long_spline[2] : short_spline[2];

  double s_start = (prev_size < 2) ? ego_car.s : previous_path.end_path_s;
  // cout << "s_start: " << s_start << endl;

  //s_start = is_free_to_go ? s_start : ego_car.s + 3;

  vector<double> next_wp0 = getXY(
      s_start + wp0_s, (2 + 4 * ego_lane),
      road.s, road.x, road.y
  );
  vector<double> next_wp1 = getXY(
      s_start + wp1_s, (2 + 4 * ego_lane),
      road.s, road.x, road.y
  );
  vector<double> next_wp2 = getXY(
      s_start + wp2_s, (2 + 4 * ego_lane),
      road.s, road.x, road.y
  );

  ptsx.push_back(next_wp0[0]);
  ptsx.push_back(next_wp1[0]);
  ptsx.push_back(next_wp2[0]);

  ptsy.push_back(next_wp0[1]);
  ptsy.push_back(next_wp1[1]);
  ptsy.push_back(next_wp2[1]);

  for ( int i = 0 ; i < ptsx.size() ; i++ ) {
    double shift_x = ptsx[i] - ref_x;
    double shift_y = ptsy[i] - ref_y;

    ptsx[i] = (shift_x * cos(0 - ref_yaw) - shift_y * sin(0 - ref_yaw));
    ptsy[i] = (shift_x * sin(0 - ref_yaw) + shift_y * cos(0 - ref_yaw));
  }

  tk::spline s_path;

  s_path.set_points(ptsx,ptsy);

//  vector<double> next_x_vals;
//  vector<double> next_y_vals;
  WayPoints x_path;
  WayPoints y_path;

  int prev_path_size_to_use = is_free_to_go ? previous_path.X.size() : 10;
  prev_path_size_to_use = previous_path.X.size();

  for ( int i = 0 ; i < prev_path_size_to_use ; i++ ) {
    x_path.push_back(previous_path.X[i]);
    y_path.push_back(previous_path.Y[i]);
  }

  cout << "x_path.size[A]:" << x_path.size() << endl;
  cout << "y_path.size[A]:" << y_path.size() << endl;

  double target_x = 30.0;
  double target_y = s_path(target_x);
  double target_dist = sqrt((target_x) * (target_x) + (target_y) * (target_y));
  double N = (target_dist / (.02 * refVelocity / 2.24) );

  double x_add_on = 0;
  //int prev_size_int = previous_path.X.size();

  for ( int i = 1 ; i <= ( 50 - prev_path_size_to_use ) ; i++ ) {
    double x_point = x_add_on + (target_x) / N;
    double y_point = s_path(x_point);

    x_add_on = x_point;

    double x_ref = x_point;
    double y_ref = y_point;

    x_point = (x_ref * cos(ref_yaw) - y_ref * sin(ref_yaw));
    y_point = (x_ref * sin(ref_yaw) + y_ref * cos(ref_yaw));

    x_point += ref_x;
    y_point += ref_y;

    //debug
    cout << "[" << i;
    cout << "] x_point:" << x_point;
    cout << ", y_point:" << y_point;
    cout << ", N:" << N;
    cout << ", x_add_on:" << x_add_on;
    cout << endl;

    x_path.push_back(x_point);
    y_path.push_back(y_point);

  }

  return Path(x_path, y_path);
}

Path SelfDrivingCar::Planner(const Path &previous_path,
                             const Vehicle &ego_car,
                             const vector<RaceCar> &racers) {
  bool two_lane_change = false;

  int ego_car_lane = road.FindLane(ego_car.d);
  double ego_car_speed = ego_car.speed * kMilesPerHourToMetersPerSecond;
  vector<double> frontcar_speeds;
  vector<double> frontcar_dists;
  for (auto lane = 0; lane <= road.kRightMostLane; ++lane) {
    auto frontcar_idx = road.FindFrontCarInLane(ego_car, racers, lane);
    double frontcar_speed = kInfinity;
    double frontcar_dist = kInfinity;
    if (frontcar_idx != -1) {
      const RaceCar & frontcar(racers[frontcar_idx]);
      frontcar_speed = sqrt(frontcar.vx*frontcar.vx + frontcar.vy*frontcar.vy);
      frontcar_dist = frontcar.s - ego_car.s;
    }
    frontcar_speeds.push_back(frontcar_speed);
    frontcar_dists.push_back(frontcar_dist);
  }

  if (frontcar_speeds[ego_car_lane] <= 0.95 * kMaxSpeed
      && frontcar_dists[ego_car_lane] <= 60) {

    int target_a = 0;
    int target_b = 2;
    if (ego_car_lane == 0) {
      target_a = 1;
    } else if (ego_car_lane == 2) {
      target_b = 1;
    }

    double cost_a = 0;
    double cost_b = 0;

    cout << "eval changing from " << ego_car_lane;
    cout << " to " << target_a;
    cout << " or to " << target_b;
    cout << endl;

    bool safe_a = SafeToChangeLane(ego_car, racers, target_a);
    bool safe_b = SafeToChangeLane(ego_car, racers, target_b);

    if (safe_a) cost_a++;
    if (safe_b) cost_b++;

    if (safe_a && frontcar_speeds[target_a] > ego_car_speed) cost_a++;
    if (safe_b && frontcar_speeds[target_b] > ego_car_speed) cost_b++;
    cout << "[compared speeds between ego and target] cost_a: " << cost_a << " , cost_b:" << cost_b << endl;

    (frontcar_speeds[target_a] > frontcar_speeds[target_b]) ? cost_a++ : cost_b++;
    cout << "[compared speeds between target lanes] cost_a: " << cost_a << " , cost_b:" << cost_b << endl;

    if (safe_a && frontcar_speeds[target_a] >= frontcar_speeds[ego_car_lane] * 1.1) cost_a++;
    if (safe_b && frontcar_speeds[target_b] >= frontcar_speeds[ego_car_lane] * 1.1) cost_b++;
    cout << "[compared speeds on target lanes] cost_a: " << cost_a << " , cost_b:" << cost_b << endl;

    if (safe_a && frontcar_dists[target_a] >= kSafeCarDistance + frontcar_dists[ego_car_lane]) cost_a++;
    if (safe_b && frontcar_dists[target_b] >= kSafeCarDistance + frontcar_dists[ego_car_lane]) cost_b++;
    cout << "[compared safe minimal distance on target lanes] cost_a: " << cost_a << " , cost_b:" << cost_b << endl;

    if (safe_a && frontcar_dists[target_a] >= (kSafeCarDistance * 3) + frontcar_dists[ego_car_lane]) cost_a++;
    if (safe_b && frontcar_dists[target_b] >= (kSafeCarDistance * 3) + frontcar_dists[ego_car_lane]) cost_b++;
    cout << "[compared long distance on target lanes] cost_a: " << cost_a << " , cost_b:" << cost_b << endl;

    if (safe_a && frontcar_dists[target_a] + kSafeCarDistance >= frontcar_dists[ego_car_lane]) cost_a++;
    if (safe_b && frontcar_dists[target_b] + kSafeCarDistance >= frontcar_dists[ego_car_lane]) cost_b++;
    cout << "[compared short distance on target lanes] cost_a: " << cost_a << " , cost_b:" << cost_b << endl;



      if (safe_a && cost_a >= 4 && cost_a > cost_b) {
        // check for two lane change
        two_lane_change =  (abs(target_a - ego_car_lane) == 2);
        cout << boolalpha;
        cout << "two_lane_change:" << two_lane_change << endl;

        if (two_lane_change) {
          cout << "its a two_lane_change..." << endl;
          if (safe_a & safe_b) {
            //refVelocity -= 3.5;
            cout << "change to lane:" << target_a << endl;
            ego_lane = target_a;
          }
        } else {
          //refVelocity -= 1.25;
          cout << "change to lane:" << target_a << endl;
          ego_lane = target_a;
        }
      }
      else if (safe_b && cost_b >= 4) {
        // check for two lane change
        two_lane_change =  (abs(target_b - ego_car_lane) == 2);
        cout << boolalpha;
        cout << "two_lane_change:" << two_lane_change << endl;

        if (two_lane_change) {
          cout << "its a two_lane_change..." << endl;
          if (safe_a & safe_b) {
            //refVelocity -= 3.5;
            cout << "change to lane:" << target_b << endl;
            ego_lane = target_b;
          }
        } else {
          //refVelocity -= 1.25;
          cout << "change to lane:" << target_b << endl;
          ego_lane = target_b;
        }
      }
    }
  return Route(previous_path, ego_car, racers, two_lane_change);
}

bool SelfDrivingCar::SafeToChangeLane(const Vehicle &ego_car,
                                      const vector<RaceCar> &racers,
                                      int target_lane) const {
  int ego_lane = road.FindLane(ego_car.d);
  int target_frontcar_idx = road.FindFrontCarInLane(ego_car, racers, target_lane);
  int target_rearcar_idx = road.FindRearCarInLane(ego_car, racers, target_lane);
  int current_frontcar_idx = road.FindFrontCarInLane(ego_car, racers, ego_lane);

  bool is_safe_to_change = (target_lane >= 0)
                           && (target_lane <= road.kRightMostLane);

  is_safe_to_change &= (ego_car.s >= 50) && (ego_car.s <= 6800);
  if (!is_safe_to_change) {
    cout << " its NOT safe to change to lane " << target_lane;
    cout << endl;
    return false;
  }

  const double ego_car_speed = ego_car.speed *kMilesPerHourToMetersPerSecond;

  if (target_frontcar_idx != -1) {
    const RaceCar & car(racers[target_frontcar_idx]);
    auto carspeed = sqrt(car.vx * car.vx + car.vy + car.vy);
    double response_time = 4; // 4 seconds
    double diff = carspeed - ego_car_speed;
    auto safe_distance = (diff >= 0) ? kSafeCarDistance :
                         fabs(diff * response_time) + kSafeCarDistance;
//        fabs(ego_car.speed * kMilesPerHourToMetersPerSecond - carspeed)
//        * response_time + kSafeCarDistance;
    auto dist = car.s - ego_car.s;

    is_safe_to_change &= (dist >= safe_distance);
    if (!is_safe_to_change) {
      cout << " its NOT safe to change to lane " << target_lane;
      cout << endl;
      return false;
    }
  }

  if (target_rearcar_idx != -1) {
    const RaceCar & car(racers[target_rearcar_idx]);
    auto carspeed = sqrt(car.vx * car.vx + car.vy * car.vy);
    double response_time = 4; // 4 seconds
    double diff = carspeed - ego_car_speed;
    auto safe_distance = (diff <= 0) ? kSafeCarDistance :
                         fabs(diff * response_time) + kSafeCarDistance;
//    auto safe_distance =
//        fabs(carspeed - ego_car.speed *kMilesPerHourToMetersPerSecond)
//        * response_time + kSafeCarDistance;
    auto dist = ego_car.s - car.s;

    is_safe_to_change &= (dist >= safe_distance);
    if (!is_safe_to_change) {
      cout << " its NOT safe to change to lane " << target_lane;
      cout << endl;
      return false;
    }
  }

  if (current_frontcar_idx != -1) {
    const RaceCar & car(racers[current_frontcar_idx]);
    auto carspeed = sqrt(car.vx * car.vx + car.vy * car.vy);
    double response_time = 3; // 2 seconds
    double diff = carspeed - ego_car_speed;
    auto safe_distance = (diff >= 0) ? kSafeCarDistance / 3 :
                         fabs(diff * response_time) + kSafeCarDistance / 3;
//    auto safe_distance =
//        fabs(ego_car.speed*kMilesPerHourToMetersPerSecond - carspeed)
//        * response_time + kSafeCarDistance;
    auto dist = car.s - ego_car.s;

    is_safe_to_change &= (dist >= safe_distance);
    if (!is_safe_to_change) {
      cout << " its NOT safe to change to lane " << target_lane;
      cout << endl;
      return false;
    }
  }

  if (!is_safe_to_change) {
    cout << " its NOT safe to change to lane " << target_lane;
    cout << endl;
    return false;
  } else {
    cout << " its SAFE to change to lane " << target_lane;
    cout << endl;
    return true;
  }
}

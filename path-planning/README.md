# CarND-Path-Planning-Project
Self-Driving Car Engineer Nanodegree Program


This one has being so far the most challenging but also the most interesting project, and I also already passed the Advance Deep Learning project. I expect that only the Capstone be more challenge than this one.

I know that there could be many improvements and I tried many of them. I had a version of this project ready for the Bosch Challenge on Sept 1st. and submitted it, but was not one of the first 25 places. Then I watched the Path Planning Walkthrough video and that was very helpful, so a good part of my code is based on what it was shown in that video.

I spent many hours trying to make the car slow down smoothly when approaching to a car on the same lane. To achieve that I started by comparing the distance of the front car with a predefined safe distance, that is adjusted with the actual speed of the ego car. If the ego car distance to the front car is smaller than that safe distance, I turn on two flags to know that the ego car needs to slow down. Those flags are `is_free_to_go` and `match_frontcar_speed`, as shown here:

```
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
    if (dist <= safe_dist) {
      is_free_to_go = false;
      if (ego_car.speed > front_car_speed) {
        match_frontcar_speed = true;
      }
    }
  }
```

Then I adjust if the speed of the ego car according to the front car speed, as shown here:

```
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
```


I also spent many hours trying to figure out the best way to change lanes smoothly, and added a cost function to evaluate if a single or a double lane change would be the best.
First, I consider the speed and distance of the car in the same lane as the ego car to know if a change lane should be considered, as shown here:

```
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
```

Then I evaluate which lane would be the best option to target. For that I evaluate first if the change to that lane is safe or not. A safe change would consider if there's enough space between the front car on that target lane, as well as a safe distance to a rear car on that lane too.

```
  int target_a = 0;
  int target_b = 2;
  if (ego_car_lane == 0) {
    target_a = 1;
  } else if (ego_car_lane == 2) {
    target_b = 1;
  }

  double cost_a = 0;
  double cost_b = 0;

  bool safe_a = SafeToChangeLane(ego_car, racers, target_a);
  bool safe_b = SafeToChangeLane(ego_car, racers, target_b);
```

After that, my cost function would add points to each target lane considering: if the front car of the target lane has greater speed than the ego car; which of the cars in the target lanes would have the greater speed; if the car in the target lane has greater speed than the front car in the ego's car lane; which car in the target lane has the longer distance or at least the minimum safe distance, as shown here:

```
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
```

And finally, depending on which of the two target lanes considered has the bigger cost, the lane of the ego car is decided.

```
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
```

So, as mentioned before, this has been the most interesting project so far. I'm really excited to jump into the Capstone project finally.

Screenshots
-----------

Click to view.

### 27 Miles

[![20 miles](/screen-shots/20-miles.png)]
[![24 miles](/screen-shots/24-miles.png)]
[![26 miles](/screen-shots/26-miles.png)]
[![27 miles](/screen-shots/27-miles.png)]



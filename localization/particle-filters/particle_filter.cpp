/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 *
 *  Updated on: May 25, 2017
 *      Author: Luis Vivero
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>

#include "particle_filter.h"

/**
 * init Initializes particle filter by initializing particles to Gaussian
 *   distribution around first position and all the weights to 1.
 * @param x Initial x position [m] (simulated estimate from GPS)
 * @param y Initial y position [m]
 * @param theta Initial orientation [rad]
 * @param std[] Array of dimension 3 [standard deviation of x [m],
 *   standard deviation of y [m]
 *   standard deviation of yaw [rad]]
 */
void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// DONE: Set the number of particles. Initialize all particles to first
	//   position (based on estimates of x, y, theta and their uncertainties
	//   from GPS) and all weights to 1.
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method
	// (and others in this file).

	// Set the number of particles.
  num_particles = 32;

  // Random number generator
  std::default_random_engine gen;

  // Normal distributions for initial coordinates
  std::normal_distribution<double> dist_x(x, std[0]); // standard deviation of x [m]
  std::normal_distribution<double> dist_y(y, std[1]); // standard deviation of y [m]
  std::normal_distribution<double> dist_theta(theta, std[2]); // standard deviation of yaw [rad]

  // Initialize all particles to first position and all weights to 1.
  for ( int i = 0 ; i < num_particles ; i++ )
  {
		Particle particle_filter;
		particle_filter.id = i;
		particle_filter.x = dist_x(gen);
		particle_filter.y = dist_y(gen);
		particle_filter.theta = dist_theta(gen);
		particle_filter.weight = 1;

    particles.push_back(particle_filter);

		// Initialize all weights to 1.
    weights.push_back(1);
  }

  // This filter has being initialized.
  is_initialized = true;
}

/**
 * prediction Predicts the state for the next time step
 *   using the process model.
 * @param delta_t Time between time step t and t+1 in measurements [s]
 * @param std_pos[] Array of dimension 3 [standard deviation of x [m],
 *   standard deviation of y [m]
 *   standard deviation of yaw [rad]]
 * @param velocity Velocity of car from t to t+1 [m/s]
 * @param yaw_rate Yaw rate of car from t to t+1 [rad/s]
 */
void ParticleFilter::prediction(double delta_t, double std_pos[],
																double velocity, double yaw_rate) {
	// DONE: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and
	// std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/

	// Random number generator
  std::default_random_engine gen;

  // Add measurements to each particle and add random Gaussian noise.
  for ( Particle& particle : particles )
  {
    // Calculate the mean for the x, y and theta.
    double mean_x = (std::abs(yaw_rate) < 0.0000001 ?
      velocity * delta_t * cos(particle.theta) :
      velocity / yaw_rate * (sin(particle.theta + yaw_rate * delta_t) - sin(particle.theta)));
    double mean_y = (std::abs(yaw_rate) < 0.0000001 ?
      velocity * delta_t * sin(particle.theta) :
      velocity / yaw_rate * (cos(particle.theta) - cos(particle.theta + yaw_rate * delta_t)));
    double mean_theta = yaw_rate * delta_t;

    // Normal distributions for new coordinates.
    std::normal_distribution<double> dist_x(particle.x + mean_x, std_pos[0]);
    std::normal_distribution<double> dist_y(particle.y + mean_y, std_pos[1]);
    std::normal_distribution<double> dist_theta(particle.theta + mean_theta, std_pos[2]);

    // Update particle with new measutements
    particle.x = dist_x(gen);
    particle.y = dist_y(gen);
    particle.theta = dist_theta(gen);
  }
}

/**
 * dataAssociation Finds which observations correspond to which landmarks
 *   (likely by using a nearest-neighbors data association).
 * @param predicted Vector of predicted landmark observations
 * @param observations Vector of landmark observations
 */
void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted,
																		 std::vector<LandmarkObs>& observations) {
	// DONE: Find the predicted measurement that is closest to each observed
	//   measurement and assign the observed measurement to this particular
	//   landmark.
	// NOTE: this method will NOT be called by the grading code. But you will
	//   probably find it useful to implement this method and use it as a
	//   helper during the updateWeights phase.

	// Iterate through observations.
	for (LandmarkObs& obj_observed : observations)
	{
		// Distance to nearest prediction.
		double nearest_distance = 100000;

		// ID of the nearest prediction.
		int nearest_id = 0;

		// Iterate through predictions.
		for (const LandmarkObs& obj_predicted : predicted)
		{
			// Compute the Euclidean distance between predicted and observation points.
			double distance = dist(obj_predicted.x, obj_predicted.y,
														 obj_observed.x, obj_observed.y);

			// Update the nearest distance.
			if (distance < nearest_distance)
			{
				nearest_distance = distance;
				nearest_id = obj_predicted.id;
			}
		}

		// Assign nearest prediction ID to current obj_observed
		obj_observed.id = nearest_id;
	}
}

/**
 * updateWeights Updates the weights for each particle based on the likelihood
 *   of the observed measurements.
 * @param sensor_range Range [m] of sensor
 * @param std_landmark[] Array of dimension 2 [standard deviation of range [m],
 *   standard deviation of bearing [rad]]
 * @param observations Vector of landmark observations
 * @param map Map class containing map landmarks
 */
void ParticleFilter::updateWeights(double sensor_range, double std_landmark[],
		std::vector<LandmarkObs> observations, Map map_landmarks) {
	// DONE: Update the weights of each particle using a mult-variate Gaussian
	//   distribution. You can read more about this distribution here:
	//   https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system.
	//   Your particles are located according to the MAP'S coordinate system.
	//   You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND
	//   translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement
	//   (look at equation 3.33. Note that you'll need to switch the minus sign in
	//   that equation to a plus to account for the fact that the map's y-axis
	//   actually points downwards.)
	//   http://planning.cs.uiuc.edu/node99.html

	// Main loop to update weights for each particle [m].
	for (int m = 0 ; m < particles.size() ; m++)
	{
		Particle& particle = particles[m];

		// Observations for particles using the MAP'S coordinate system.
		std::vector<LandmarkObs> observations_maps;

		const float cos_theta = cos(particle.theta);
		const float sin_theta = sin(particle.theta);

		// Calculate and add each observation into the observations landmark vector.
		for (const LandmarkObs& observation : observations)
		{
			// Translate and rotate each landmark from vehicle coordinate system to
			// map's coordinate system
			LandmarkObs landmark;
			landmark.id = observation.id;
			landmark.x = observation.x * cos_theta -
									   observation.y * sin_theta + particle.x;
			landmark.y = observation.x * sin_theta +
									   observation.y * cos_theta + particle.y;
			observations_maps.push_back(landmark);
		}

		// Predicted landmark vector.
		std::vector<LandmarkObs> predicted;

		// Iterate the Map to populate the prediction landmark vector.
		for (const Map::single_landmark_s& landmark : map_landmarks.landmark_list)
		{
			// Verify if landmark is within sensor range.
			if (dist(landmark.x_f, landmark.y_f, particle.x, particle.y) <= sensor_range)
			{
				// Create the prediction object.
				LandmarkObs prediction {landmark.id_i, landmark.x_f, landmark.y_f};
				predicted.push_back(prediction);
			}
		}

		// Find which observations correspond to which landmarks.
		dataAssociation(predicted, observations_maps);

		// The cummulative_weight is used to combine the likelihoods of all measurements.
		double cummulative_weight = 1;

		// Iterate through each observation
		for (const LandmarkObs& observation : observations_maps)
		{
			// Assumption: index of landmark in a map is equal to id of landmark - 1
			Map::single_landmark_s prediction =
																map_landmarks.landmark_list[observation.id - 1];

			// Difference between observed and predicted measurements.
			double dx = observation.x - prediction.x_f;
			double dy = observation.y - prediction.y_f;

			// Update the weight of the particle using the multivaritate Gaussian
			// probability density function of this measurement.
			double weight = 1 / (2 * M_PI * std_landmark[0] * std_landmark[1])
			  * std::exp(-1 * (pow(dx, 2) / pow(std_landmark[0], 2)
				+ pow(dy, 2) / pow(std_landmark[1], 2)));

			// Update the cummulative weight.
			cummulative_weight *= weight;
		}

		// Assign the new weight to the particle
		particle.weight = cummulative_weight;

		// Assign weight to the list of weights
		weights[m] = cummulative_weight;
	}
}

/**
 * resample Resamples from the updated set of particles to form
 *   the new set of particles.
 */
void ParticleFilter::resample() {
	// DONE: Resample particles with replacement with probability proportional to
	//   their weight.
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution

	// Create a randome engine generator.
  std::default_random_engine gen;

	// Create a discrete distribution.
  std::discrete_distribution<int> distribution {weights.begin(), weights.end()};

  // Resampled particles vector.
  std::vector<Particle> resampled_particles;

  // Resample particles with probability proportional to their weight.
  for (int i = 0; i < num_particles; i++) {
    // Generate particle index using the distribution.
    int index = distribution(gen);

    // Get a particle from the list of particles.
    Particle particle = particles[index];

    // Push back the particle to the resampled vector.
    resampled_particles.push_back(particle);
  }

  particles = resampled_particles;
}

void ParticleFilter::write(std::string filename) {
	// You don't need to modify this file.
	std::ofstream dataFile;
	dataFile.open(filename, std::ios::app);
	for (int i = 0; i < num_particles; ++i) {
		dataFile << particles[i].x << " " << particles[i].y << " " << particles[i].theta << "\n";
	}
	dataFile.close();
}

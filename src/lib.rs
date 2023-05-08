//! A proportional-integral-derivative (PID) controller.
#![no_std]

use core::cmp::PartialOrd;
use core::ops::{Add, Mul, Neg, Sub};
use num_traits::Zero;

pub struct Pid<T>
where
    T: PartialOrd + Copy + Clone,
{
    /// Proportional gain.
    kp: T,
    /// Integral gain.
    ki: T,
    /// Derivative gain.
    kd: T,
    /// Feed forward gain.
    kff: T,
    /// Limit of contribution of P term
    p_limit: Option<Limit<T>>,
    /// Limit of contribution of I term
    i_limit: Option<Limit<T>>,
    /// Limit of contribution of D term
    d_limit: Option<Limit<T>>,
    /// Limit of contribution of feed forward term
    ff_limit: Option<Limit<T>>,
    /// Limit of output
    output_limit: Option<Limit<T>>,

    setpoint: T,
    prev_measurement: Option<T>,
    /// `integral_term = sum[error(t) * ki(t)] (for all t)`
    integral_term: T,
    direction: Direction,
}

pub struct ControlOutput<T> {
    /// Contribution of the P term to the output.
    pub p: T,
    /// Contribution of the I term to the output.
    /// `i = sum[error(t) * ki(t)] (for all t)`
    pub i: T,
    /// Contribution of the D term to the output.
    pub d: T,
    /// Contribution of the feed forward term to the output.
    pub ff: T,
    /// Output of the PID controller.
    pub output: T,
}

impl<T> Pid<T>
where
    T: Add<Output = T>
        + Sub<Output = T>
        + Neg<Output = T>
        + Mul<Output = T>
        + Zero
        + Default
        + PartialOrd
        + Copy
        + Clone,
{
    pub fn new(kp: T, ki: T, kd: T, kff: T, setpoint: T) -> Self {
        Self {
            kp,
            ki,
            kd,
            kff,
            p_limit: None,
            i_limit: None,
            d_limit: None,
            ff_limit: None,
            output_limit: None,
            setpoint,
            prev_measurement: None,
            integral_term: T::zero(),
            direction: Direction::Direct,
        }
    }

    pub fn set_limits(
        &mut self,
        p_limit: Option<Limit<T>>,
        i_limit: Option<Limit<T>>,
        d_limit: Option<Limit<T>>,
        ff_limit: Option<Limit<T>>,
        output_limit: Option<Limit<T>>,
    ) {
        self.p_limit = p_limit;
        self.i_limit = i_limit;
        self.d_limit = d_limit;
        self.ff_limit = ff_limit;
        self.output_limit = output_limit;
    }

    pub fn update_pid_terms(&mut self, kp: T, ki: T, kd: T, kff: T) {
        self.kp = kp;
        self.ki = ki;
        self.kd = kd;
        self.kff = kff;
    }

    pub fn get_setpoint(&mut self) -> T {
        self.setpoint
    }

    pub fn update_setpoint(&mut self, setpoint: T) {
        self.setpoint = setpoint;
    }

    pub fn set_direction(&mut self, direction: Direction) {
        self.direction = direction;
    }

    /// Resets the integral term back to zero. This may drastically change the
    /// control output.
    pub fn reset_integral_term(&mut self) {
        self.integral_term = T::zero();
    }

    /// Given a new measurement, calculates the next control output.
    ///
    /// # Panics
    /// If a setpoint has not been set via `update_setpoint()`.
    pub fn next_control_output(&mut self, measurement: T) -> ControlOutput<T> {
        let error = self.setpoint - measurement;

        let p_unbounded = match self.direction {
            Direction::Direct => error * self.kp,
            Direction::Reverse => error * -self.kp,
        };
        let p = match &self.p_limit {
            None => p_unbounded,
            Some(limit) => limit.apply(p_unbounded),
        };

        // Mitigate output jumps when ki(t) != ki(t-1).
        // While it's standard to use an error_integral that's a running sum of
        // just the error (no ki), because we support ki changing dynamically,
        // we store the entire term so that we don't need to remember previous
        // ki values.
        self.integral_term = match self.direction {
            Direction::Direct => self.integral_term + error * self.ki,
            Direction::Reverse => self.integral_term + error * -self.ki,
        };
        // Mitigate integral windup: Don't want to keep building up error
        // beyond what i_limit will allow.
        self.integral_term = match &self.i_limit {
            None => self.integral_term,
            Some(limit) => limit.apply(self.integral_term),
        };

        // Mitigate derivative kick: Use the derivative of the measurement
        // rather than the derivative of the error.
        let d_unbounded = -match self.prev_measurement.as_ref() {
            Some(prev_measurement) => measurement - *prev_measurement,
            None => T::zero(),
        } * match self.direction {
            Direction::Direct => self.kd,
            Direction::Reverse => -self.kd,
        };
        self.prev_measurement = Some(measurement);
        let d = match &self.d_limit {
            None => d_unbounded,
            Some(limit) => limit.apply(d_unbounded),
        };

        let ff_unbounded = match self.direction {
            Direction::Direct => self.setpoint * self.kff,
            Direction::Reverse => self.setpoint * -self.kff,
        };
        let ff = match &self.ff_limit {
            None => ff_unbounded,
            Some(limit) => limit.apply(ff_unbounded),
        };

        let output_unbounded = p + self.integral_term + d + ff;
        let output = match &self.output_limit {
            None => output_unbounded,
            Some(limit) => limit.apply(output_unbounded),
        };

        ControlOutput {
            p,
            i: self.integral_term,
            d,
            ff,
            output,
        }
    }
}

#[derive(Clone, Copy)]
pub struct Limit<T>
where
    T: PartialOrd + Copy + Clone,
{
    min: T,
    max: T,
}

impl<T> Limit<T>
where
    T: PartialOrd + Copy + Clone,
{
    pub fn new(min: T, max: T) -> Self {
        Self { min, max }
    }

    pub fn apply(&self, value: T) -> T {
        if value > self.max {
            self.max
        } else if value < self.min {
            self.min
        } else {
            value
        }
    }
}

pub enum Direction {
    Direct,
    Reverse,
}

#[cfg(test)]
mod tests {
    use crate::Limit;

    use super::Pid;

    #[test]
    fn proportional() {
        let mut pid = Pid::new(2.0, 0.0, 0.0, 10.0);
        assert_eq!(pid.setpoint, 10.0);

        // Test simple proportional
        assert_eq!(pid.next_control_output(0.0).output, 20.0);

        // Test proportional limit
        pid.set_limits(Some(Limit::new(-10.0, 10.0)), None, None, None);
        assert_eq!(pid.next_control_output(0.0).output, 10.0);
    }

    #[test]
    fn derivative() {
        let mut pid = Pid::new(0.0, 0.0, 2.0, 10.0);

        // Test that there's no derivative since it's the first measurement
        assert_eq!(pid.next_control_output(0.0).output, 0.0);

        // Test that there's now a derivative
        assert_eq!(pid.next_control_output(5.0).output, -10.0);

        // Test derivative limit
        pid.set_limits(None, None, Some(Limit::new(-5.0, 5.0)), None);
        assert_eq!(pid.next_control_output(10.0).output, -5.0);
    }

    #[test]
    fn integral() {
        let mut pid = Pid::new(0.0, 2.0, 0.0, 10.0);

        // Test basic integration
        assert_eq!(pid.next_control_output(0.0).output, 20.0);
        assert_eq!(pid.next_control_output(0.0).output, 40.0);
        assert_eq!(pid.next_control_output(5.0).output, 50.0);

        // Test limit
        pid.set_limits(None, Some(Limit::new(-50.0, 50.0)), None, None);
        assert_eq!(pid.next_control_output(5.0).output, 50.0);
        // Test that limit doesn't impede reversal of error integral
        assert_eq!(pid.next_control_output(15.0).output, 40.0);

        // Test that error integral accumulates negative values
        let mut pid2 = Pid::new(0.0, 2.0, 0.0, -10.0);
        assert_eq!(pid2.next_control_output(0.0).output, -20.0);
        assert_eq!(pid2.next_control_output(0.0).output, -40.0);

        pid2.set_limits(None, Some(Limit::new(-50.0, 50.0)), None, None);
        assert_eq!(pid2.next_control_output(-5.0).output, -50.0);
        // Test that limit doesn't impede reversal of error integral
        assert_eq!(pid2.next_control_output(-15.0).output, -40.0);
    }

    #[test]
    fn output_limit() {
        let mut pid = Pid::new(1.0, 0.0, 0.0, 10.0);
        pid.set_limits(None, None, None, Some(Limit::new(-1.0, 1.0)));

        let out = pid.next_control_output(0.0);
        assert_eq!(out.p, 10.0); // 1.0 * 10.0
        assert_eq!(out.output, 1.0);

        let out = pid.next_control_output(20.0);
        assert_eq!(out.p, -10.0); // 1.0 * (10.0 - 20.0)
        assert_eq!(out.output, -1.0);
    }

    #[test]
    fn pid() {
        let mut pid = Pid::new(1.0, 0.1, 1.0, 10.0);

        let out = pid.next_control_output(0.0);
        assert_eq!(out.p, 10.0); // 1.0 * 10.0
        assert_eq!(out.i, 1.0); // 0.1 * 10.0
        assert_eq!(out.d, 0.0); // -(1.0 * 0.0)
        assert_eq!(out.output, 11.0);

        let out = pid.next_control_output(5.0);
        assert_eq!(out.p, 5.0); // 1.0 * 5.0
        assert_eq!(out.i, 1.5); // 0.1 * (10.0 + 5.0)
        assert_eq!(out.d, -5.0); // -(1.0 * 5.0)
        assert_eq!(out.output, 1.5);

        let out = pid.next_control_output(11.0);
        assert_eq!(out.p, -1.0); // 1.0 * -1.0
        assert_eq!(out.i, 1.4); // 0.1 * (10.0 + 5.0 - 1)
        assert_eq!(out.d, -6.0); // -(1.0 * 6.0)
        assert_eq!(out.output, -5.6);

        let out = pid.next_control_output(10.0);
        assert_eq!(out.p, 0.0); // 1.0 * 0.0
        assert_eq!(out.i, 1.4); // 0.1 * (10.0 + 5.0 - 1.0 + 0.0)
        assert_eq!(out.d, 1.0); // -(1.0 * -1.0)
        assert_eq!(out.output, 2.4);
    }

    #[test]
    fn f32_and_f64() {
        let mut pid32 = Pid::new(2.0f32, 0.0, 0.0, 10.0);

        let mut pid64 = Pid::new(2.0f64, 0.0, 0.0, 10.0);

        assert_eq!(
            pid32.next_control_output(0.0).output,
            pid64.next_control_output(0.0).output as f32
        );
        assert_eq!(
            pid32.next_control_output(0.0).output as f64,
            pid64.next_control_output(0.0).output
        );
    }
}

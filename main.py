from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field
from ortools.sat.python import cp_model
from datetime import datetime, timedelta
from typing import List, Dict, Any
from pathlib import Path
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import math

app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/", response_class=HTMLResponse)
async def serve_frontend(request: Request):
    index_path = Path(__file__).parent / "static" / "index.html"
    with open(index_path, encoding='utf-8') as f:
        return HTMLResponse(content=f.read(), status_code=200)

# Add new endpoint to handle schedule updates
@app.put("/api/update-schedule")
async def update_schedule(update_data: dict):
    try:
        # Validate and process updates
        employee = update_data['employee']
        date = update_data['date']
        new_shift = update_data['shift']
        
        # Validate required fields
        if not employee or not date:
            raise ValueError("Employee and date are required fields")
        
        # Validate shift data
        if isinstance(new_shift, list):
            for shift in new_shift:
                if isinstance(shift, dict) and 'shift' in shift and 'task' in shift:
                    # Validate shift times
                    if 'start' not in shift or 'end' not in shift:
                        raise ValueError(f"Start and end times are required for shift {shift.get('shift')}")
                    
                    # Validate duration
                    if 'duration_hours' not in shift:
                        # Calculate duration if not provided
                        try:
                            start_time = datetime.strptime(shift['start'], "%H:%M").time()
                            end_time = datetime.strptime(shift['end'], "%H:%M").time()
                            duration = (datetime.combine(datetime.min, end_time) - 
                                      datetime.combine(datetime.min, start_time)).seconds // 3600
                            shift['duration_hours'] = duration
                        except ValueError:
                            raise ValueError(f"Invalid time format in shift {shift.get('shift')}")
        
        # In a real implementation, you would:
        # 1. Retrieve the current schedule from a database
        # 2. Update the specific employee's schedule for the given date
        # 3. Validate the changes against constraints (optional)
        # 4. Save the updated schedule back to the database
        
        # For now, we'll just return success
        return {
            "status": "success", 
            "message": "Schedule updated successfully",
            "employee": employee,
            "date": date,
            "updated_shift": new_shift
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")
    

class SolverRequest(BaseModel):
    employees: List[Dict[str, Any]] = Field(..., description="List of employees with their requests")
    shifts: List[Dict[str, Any]]
    closed_days: List[str]
    start_date: str
    schedule_days: int

class Violation:
    def __init__(self, type: str, message: str, employees: List[str] = None, dates: List[str] = None, details: Dict = None):
        self.type = type
        self.message = message
        self.employees = employees or []
        self.dates = dates or []
        self.details = details or {}

    def to_dict(self):
        return {
            "type": self.type,
            "message": self.message,
            "employees": self.employees,
            "dates": self.dates,
            "details": self.details
        }

class ImprovedScheduleGenerator:
    def __init__(self, request: SolverRequest):
        self.request = request
        self.model = cp_model.CpModel()
        self.all_employees = []
        self.employee_details = {}
        self.shift_structure = {}
        self.task_durations = {}
        self.assignments = {}
        self.penalties = []
        self.total_hours_vars = {}
        self.employee_working_vars = {}
        self.employee_shift_work_vars = {}
        self.employee_double_shifts = {}
        self.violations = []  # Track violations

        # Constants for penalties (taken from colab.py)
        self.PENALTY_WEIGHTS = {
            'supervisor': 1000,
            'consecutive': 1000,
            'preference': 800,
            'hours': 1000,
            'fairness': 300,
            'two_consecutive_off': 900,
            'one_shift': 900,
            'consecutive_double': 1000,  # Strong penalty for consecutive double shifts
        }
        self.CONSECUTIVE_OFF_REWARD = 300
        self.CONSECUTIVE_OFF_PENALTY = self.PENALTY_WEIGHTS['two_consecutive_off']

        try:
            # Parse start date in UTC
            self.start_date = datetime.strptime(request.start_date, "%Y-%m-%d").date()
            self.days = [
                self.start_date + timedelta(days=i) 
                for i in range(request.schedule_days)
            ]
            # Add validation for unique dates
            unique_days = len({d.isoformat() for d in self.days})
            if unique_days != request.schedule_days:
                raise ValueError("Date sequence contains duplicates")
            
            self.active_days = [day for day in self.days if day.isoformat() not in request.closed_days]
            self.num_full_weeks = len(self.days) // 7
            self._prepare_data_structures()
        except Exception as e:
            raise ValueError(f"Data validation failed: {str(e)}")

    def _prepare_data_structures(self):
        if not self.request.employees:
            raise ValueError("At least one employee required")
        if not self.request.shifts:
            raise ValueError("At least one shift required")

        for emp in self.request.employees:
            self.all_employees.append(emp['name'])
            
            skills = emp.get('skills', [])
            if isinstance(skills, str):
                skills = [skills]

            # Separate vacation days from other time off
            vacation_days = set()
            time_off_days = set()
            
            for request in emp.get('requests', []):
                if request.get('status') == 'Approved':
                    if request['type'] == 'Vacation Leave':
                        for date_str in request.get('dates', []):
                            try:
                                date = datetime.strptime(date_str, "%Y-%m-%d").date()
                                if self.start_date <= date < (self.start_date + timedelta(days=self.request.schedule_days)):
                                    vacation_days.add(date.isoformat())
                            except ValueError:
                                continue
                    elif request['type'] == 'Time Off':
                        for date_str in request.get('dates', []):
                            try:
                                date = datetime.strptime(date_str, "%Y-%m-%d").date()
                                if self.start_date <= date < (self.start_date + timedelta(days=self.request.schedule_days)):
                                    time_off_days.add(date.isoformat())
                            except ValueError:
                                continue

            # Handle shift preferences
            preferred_shifts = {}
            for request in emp.get('requests', []):
                if request.get('type') == 'Shift Preference' and request.get('status') == 'Approved':
                    for date_str in request.get('dates', []):
                        preferred_shifts[date_str] = request.get('shiftPreference')

            # Get contract parameters
            contract = emp.get('contract', {})
            min_hours = contract.get('minHours', 0)
            max_hours = contract.get('maxHours', 168)
            
            self.employee_details[emp['name']] = {
                'skills': set(skills),
                'position': emp.get('position', 'Employee').lower(),
                'contract': contract,
                'vacation_days': vacation_days,
                'time_off_days': time_off_days,
                'available_days': [
                    day.isoformat() for day in self.days 
                    if day.isoformat() not in self.request.closed_days 
                    and day.isoformat() not in vacation_days
                    and day.isoformat() not in time_off_days
                ],
                'preferred_shifts_by_date': preferred_shifts,
                'days_off': vacation_days.union(time_off_days),
                'requests': emp.get('requests', [])
            }

        # Process shifts and tasks
        for shift in self.request.shifts:
            shift_name = shift['name']
            tasks_info = []
            for task in shift.get('tasks', []):
                task_schedule = {}
                for day_sched in task.get('schedule', []):
                    try:
                        start = datetime.strptime(day_sched.get('startTime', '00:00'), "%H:%M").time()
                        end = datetime.strptime(day_sched.get('endTime', '00:00'), "%H:%M").time()
                        duration = (datetime.combine(datetime.min, end) - 
                                  datetime.combine(datetime.min, start)).seconds // 3600
                        # Store duration per weekday
                        task_schedule[day_sched['day']] = duration
                    except ValueError as e:
                        raise ValueError(f"Invalid time format in {shift_name} - {task.get('name', 'unnamed-task')}: {str(e)}")

                skill_requirement = task.get('skillRequirement') or task.get('skill_requirement')
                if not skill_requirement:
                    raise ValueError(
                        f"Missing skill requirement in shift '{shift_name}' "
                        f"for task '{task.get('name', 'unnamed-task')}'"
                    )

                tasks_info.append({
                    'name': task['name'],
                    'skill_requirement': skill_requirement,
                    'schedule': task_schedule
                })
                
                # Map task durations for all days in the schedule
                for day in self.days:
                    day_str = day.isoformat()
                    weekday = day.strftime('%A')
                    if weekday in task_schedule:
                        task_key = (shift_name, task['name'], day_str)
                        self.task_durations[task_key] = task_schedule[weekday]
                    else:
                        # Explicitly set to 0 for unscheduled days
                        task_key = (shift_name, task['name'], day_str)
                        self.task_durations[task_key] = 0

            self.shift_structure[shift_name] = {
                'tasks': tasks_info,
                'required_skills': {task['name']: task['skill_requirement'] for task in tasks_info}
            }

    def _create_assignment_variables(self):
        self.assignments.clear()
        
        # First create all possible assignments
        for emp in self.all_employees:
            emp_info = self.employee_details[emp]
            for day in self.days:
                day_str = day.isoformat()
                if day_str in self.request.closed_days or day_str in emp_info['days_off']:
                    continue
                    
                for shift_name, shift_data in self.shift_structure.items():
                    for task in shift_data['tasks']:
                        if task['skill_requirement'] in emp_info['skills']:
                            # Check if the task is scheduled on this day
                            duration = self.task_durations.get((shift_name, task['name'], day_str), 0)
                            if duration > 0:
                                key = (emp, day_str, shift_name, task['name'])
                                var = self.model.NewBoolVar(f'assign_{emp}_{day_str}_{shift_name}_{task["name"]}')
                                self.assignments[key] = (var, duration)

        # Enhanced preference handling
        for emp in self.all_employees:
            emp_info = self.employee_details[emp]
            
            # Force days off first
            for date_str in emp_info['days_off']:
                self._force_day_off(emp, date_str)

            # Then handle shift preferences
            preferred_shifts = emp_info.get('preferred_shifts_by_date', {})
            for date_str, preferred_shift in preferred_shifts.items():
                # Validate date is within schedule
                try:
                    if datetime.strptime(date_str, "%Y-%m-%d").date() not in self.days:
                        continue
                except ValueError:
                    continue
                
                # Get all possible assignments for this day
                all_assignments = [
                    var for (e, d, s, t), (var, _) in self.assignments.items()
                    if e == emp and d == date_str
                ]
                
                # If employee should work this day, enforce preference
                if date_str in emp_info['available_days']:
                    preferred_vars = [
                        var for (e, d, s, t), (var, _) in self.assignments.items()
                        if e == emp and d == date_str and s == preferred_shift
                    ]
                    
                    if preferred_vars:
                        # Must work preferred shift and no others
                        self.model.Add(sum(preferred_vars) >= 1)
                        self.model.Add(sum(all_assignments) == sum(preferred_vars))
                    else:
                        # If preferred shift isn't available, force day off
                        self.model.Add(sum(all_assignments) == 0)

    def _force_day_off(self, emp, date_str):
        """Force employee to have day off by setting all assignments to 0"""
        for key in list(self.assignments.keys()):
            if key[0] == emp and key[1] == date_str:
                var, _ = self.assignments[key]
                self.model.Add(var == 0)


    def _create_working_day_indicators(self):
        # Create working day indicators for each employee (similar to colab.py)
        for emp in self.all_employees:
            self.employee_working_vars[emp] = {}
            for day in self.days:
                day_str = day.isoformat()
                if day_str in self.request.closed_days or day_str in self.employee_details[emp]['days_off']:
                    continue
                    
                working = self.model.NewBoolVar(f'work_{emp}_{day_str}')
                shifts = [var for (e, d, s, t), (var, _) in self.assignments.items()
                         if e == emp and d == day_str]
                
                if shifts:  # Only add constraints if there are shifts available
                    self.model.Add(sum(shifts) >= 1).OnlyEnforceIf(working)
                    self.model.Add(sum(shifts) == 0).OnlyEnforceIf(working.Not())
                    self.employee_working_vars[emp][day_str] = working

    def _create_shift_indicators(self):
        # Create shift indicators for each employee and shift
        for emp in self.all_employees:
            for day in self.active_days:
                day_str = day.isoformat()
                if day_str in self.request.closed_days or day_str in self.employee_details[emp]['days_off']:
                    continue
                    
                for shift_name in self.shift_structure:
                    tasks = [var for (e, d, s, t), (var, _) in self.assignments.items()
                           if e == emp and d == day_str and s == shift_name]
                    if tasks:
                        work_shift = self.model.NewBoolVar(f'work_{emp}_{day_str}_{shift_name}')
                        self.model.Add(sum(tasks) >= 1).OnlyEnforceIf(work_shift)
                        self.model.Add(sum(tasks) == 0).OnlyEnforceIf(work_shift.Not())
                        self.employee_shift_work_vars[(emp, day_str, shift_name)] = work_shift

    def _create_double_shift_indicators(self):
    # Dynamically get the first two shifts as potential double shifts
        shift_names = list(self.shift_structure.keys())
        if len(shift_names) < 2:
            return  # Not enough shifts for double shifts

        first_shift, second_shift = shift_names[:2]

        # Track double shifts
        for emp in self.all_employees:
            self.employee_double_shifts[emp] = {}
            for day in self.active_days:
                day_str = day.isoformat()
                if day_str in self.request.closed_days or day_str in self.employee_details[emp]['days_off']:
                    continue
                
                first_var = self.employee_shift_work_vars.get((emp, day_str, first_shift))
                second_var = self.employee_shift_work_vars.get((emp, day_str, second_shift))
                
                if first_var is not None and second_var is not None:
                    double_shift = self.model.NewBoolVar(f'double_{emp}_{day_str}')
                    self.model.AddBoolAnd([first_var, second_var]).OnlyEnforceIf(double_shift)
                    self.model.AddBoolOr([first_var.Not(), second_var.Not()]).OnlyEnforceIf(double_shift.Not())
                    self.employee_double_shifts[emp][day_str] = double_shift

    def _add_task_coverage_constraint(self):
        # Ensure each task is assigned exactly one employee
        for day in self.days:
            day_str = day.isoformat()
            if day_str in self.request.closed_days:
                continue
                
            for shift_name, shift_data in self.shift_structure.items():
                for task in shift_data['tasks']:
                    task_name = task['name']
                    
                    qualified_vars = [
                        var for (emp, d, s, t), (var, _) in self.assignments.items()
                        if d == day_str and s == shift_name and t == task_name
                    ]
                    
                    if qualified_vars:
                        self.model.AddExactlyOne(qualified_vars)

    def _add_one_task_per_employee_constraint(self):
        # An employee can only be assigned one task per shift
        for emp in self.all_employees:
            for day in self.days:
                day_str = day.isoformat()
                if day_str in self.request.closed_days or day_str in self.employee_details[emp]['days_off']:
                    continue
                    
                for shift_name in self.shift_structure:
                    task_vars = [
                        var for (e, d, s, t), (var, _) in self.assignments.items()
                        if e == emp and d == day_str and s == shift_name
                    ]
                    if task_vars:
                        self.model.AddAtMostOne(task_vars)

    def _add_supervisor_constraint(self):
        # Ensure at least one supervisor per shift
        for day in self.active_days:
            day_str = day.isoformat()
            if day_str in self.request.closed_days:
                continue
                
            for shift_name in self.shift_structure:
                supervisor_vars = []
                for emp in self.all_employees:
                    # Check if employee is a supervisor
                    if self.employee_details[emp]['position'] == 'supervisor':
                        for task_name in self.shift_structure[shift_name]['required_skills']:
                            required_skill = self.shift_structure[shift_name]['required_skills'][task_name]
                            if required_skill in self.employee_details[emp]['skills']:
                                key = (emp, day_str, shift_name, task_name)
                                if key in self.assignments:
                                    supervisor_vars.append(self.assignments[key][0])
                
                if supervisor_vars:
                    # At least one supervisor must be assigned to this shift
                    self.model.AddBoolOr(supervisor_vars)
                else:
                    # If no supervisors are available, we shouldn't create an infeasible problem
                    # Instead, we'll make this a soft constraint with a high penalty
                    has_supervisor = self.model.NewBoolVar(f'has_supervisor_{day_str}_{shift_name}')
                    self.model.Add(has_supervisor == 0)  # Always false since no supervisors
                    self.penalties.append(has_supervisor.Not() * self.PENALTY_WEIGHTS['supervisor'])
                    
                    # Add violation record
                    self.violations.append(Violation(
                        type="SUPERVISOR",
                        message=f"No supervisor available for shift {shift_name} on {day_str}",
                        dates=[day_str],
                        details={"shift": shift_name}
                    ))

    def _add_max_consecutive_days_constraint(self):
        # Maximum 5 consecutive working days
        for emp in self.all_employees:
            working_days = []
            for day in self.days:
                day_str = day.isoformat()
                if day_str in self.request.closed_days or day_str in self.employee_details[emp]['days_off']:
                    working_days.append(0)
                else:
                    working_var = self.employee_working_vars.get(emp, {}).get(day_str)
                    if working_var is not None:
                        working_days.append(working_var)
                    else:
                        working_days.append(0)
            
            # Check consecutive working days
            for i in range(len(working_days) - 5):
                window = working_days[i:i+6]
                var_window = [var for var in window if not isinstance(var, int)]
                
                if len(var_window) > 5:
                    self.model.Add(sum(var_window) <= 5)
                    
                    # Add violation record for potential consecutive days
                    dates = [self.days[i+j].isoformat() for j in range(6)]
                    self.violations.append(Violation(
                        type="CONSECUTIVE_DAYS",
                        message=f"Employee {emp} may work more than 5 consecutive days",
                        employees=[emp],
                        dates=dates,
                        details={"max_consecutive": 5}
                    ))

    def _add_two_days_off_per_week_constraint(self):
        # At least 2 days off per week
        for emp in self.all_employees:
            for week in range(self.num_full_weeks):
                week_days = self.days[week*7:(week+1)*7]
                off_terms = []
                
                for day in week_days:
                    day_str = day.isoformat()
                    if day_str in self.request.closed_days or day_str in self.employee_details[emp]['days_off']:
                        # Definitely off
                        off_terms.append(1)
                    else:
                        working_var = self.employee_working_vars.get(emp, {}).get(day_str)
                        if working_var is not None:
                            # 1 - working_var = off indicator
                            off_var = self.model.NewIntVar(0, 1, f'off_{emp}_{day_str}')
                            self.model.Add(off_var == 1 - working_var)
                            off_terms.append(off_var)
                        else:
                            # If no working variable exists, consider as day off
                            off_terms.append(1)
                
                # Add constraint only if meaningful
                if off_terms:
                    self.model.Add(sum(off_terms) >= 2)

    def _get_week_boundaries(self):
        """Get the start and end dates for each week in the schedule period"""
        weeks = []
        current_date = self.start_date
        
        # Find the first Monday
        while current_date.strftime('%A') != 'Monday':
            current_date += timedelta(days=1)
        
        # Create week boundaries
        while current_date < self.start_date + timedelta(days=self.request.schedule_days):
            week_end = current_date + timedelta(days=6)
            weeks.append((current_date, week_end))
            current_date += timedelta(days=7)
        
        return weeks

    def _add_contract_hours_constraint(self):
        week_boundaries = self._get_week_boundaries()
        
        for emp in self.all_employees:
            emp_info = self.employee_details[emp]
            contract = emp_info['contract']
            original_min = int(contract.get('minHours', 0))
            original_max = int(contract.get('maxHours', 168))
            
            # Calculate total days in schedule
            total_days = len(self.days)
            
            # Calculate total days off (both vacation and time off)
            total_days_off = sum(1 for day in self.days 
                               if day.isoformat() in emp_info['vacation_days'] or 
                               day.isoformat() in emp_info['time_off_days'])
            
            # Calculate actual working days (excluding closed days and all days off)
            working_days = sum(1 for day in self.days 
                             if day.isoformat() not in self.request.closed_days 
                             and day.isoformat() not in emp_info['vacation_days']
                             and day.isoformat() not in emp_info['time_off_days'])
            
            # Calculate adjusted contract hours based on working days
            # Only adjust if there are days off or closed days
            if total_days_off > 0 or len(self.request.closed_days) > 0:
                adjusted_min = math.floor(original_min * (working_days / total_days))
                adjusted_max = math.ceil(original_max * (working_days / total_days))
            else:
                adjusted_min = original_min
                adjusted_max = original_max
            
            # Create variables for total hours
            total_hours = self.model.NewIntVar(0, 168*2, f'total_hours_{emp}')
            
            # Sum up all assignments for this employee
            assignment_terms = []
            for key, (var, duration) in self.assignments.items():
                e, _, _, _ = key
                if e == emp:
                    assignment_terms.append(var * duration)
            
            if assignment_terms:
                self.model.Add(total_hours == sum(assignment_terms))
            
            # Add soft constraints with penalties
            under = self.model.NewIntVar(0, 168*2, f'under_{emp}')
            over = self.model.NewIntVar(0, 168*2, f'over_{emp}')
            
            self.model.Add(under >= adjusted_min - total_hours)
            self.model.Add(over >= total_hours - adjusted_max)
            
            self.penalties.append(under * self.PENALTY_WEIGHTS['hours'])
            self.penalties.append(over * self.PENALTY_WEIGHTS['hours'])
            
            # Store the total hours variable for later use
            self.total_hours_vars[emp] = total_hours

    def _add_fairness_constraint(self):
        # Add fairness constraint to balance hours between employees
        if len(self.total_hours_vars) >= 2:  # Only add if we have at least 2 employees
            total_hours_list = list(self.total_hours_vars.values())
            max_h = self.model.NewIntVar(0, 168*2, 'max_hours')
            min_h = self.model.NewIntVar(0, 168*2, 'min_hours')
            self.model.AddMaxEquality(max_h, total_hours_list)
            self.model.AddMinEquality(min_h, total_hours_list)
            diff = self.model.NewIntVar(0, 168*2, 'hours_diff')
            self.model.Add(diff == max_h - min_h)
            self.penalties.append(diff * self.PENALTY_WEIGHTS['fairness'])

    def _add_consecutive_off_days_reward(self):
        # Reward consecutive days off
        for emp in self.all_employees:
            for week in range(self.num_full_weeks):
                week_days = self.days[week*7:(week+1)*7]
                off_indicators = []
                
                for day in week_days:
                    day_str = day.isoformat()
                    if day_str in self.request.closed_days or day_str in self.employee_details[emp]['days_off']:
                        off_indicators.append(1)  # Definitely off
                    else:
                        working_var = self.employee_working_vars.get(emp, {}).get(day_str)
                        if working_var is not None:
                            off_var = self.model.NewIntVar(0, 1, f'off_{emp}_{day_str}')
                            self.model.Add(off_var == 1 - working_var)
                            off_indicators.append(off_var)
                        else:
                            off_indicators.append(1)  # No working variable means off
                
                # Check for consecutive days off
                if len(off_indicators) >= 2:
                    pair_vars = []
                    for i in range(len(off_indicators) - 1):
                        pair_off = self.model.NewBoolVar(f'pair_off_{emp}_week{week}_{i}')
                        
                        # Handle both constant and variable cases
                        if isinstance(off_indicators[i], int) and isinstance(off_indicators[i+1], int):
                            if off_indicators[i] == 1 and off_indicators[i+1] == 1:
                                self.model.Add(pair_off == 1)
                            else:
                                self.model.Add(pair_off == 0)
                        elif isinstance(off_indicators[i], int):
                            if off_indicators[i] == 1:
                                self.model.Add(pair_off == off_indicators[i+1])
                            else:
                                self.model.Add(pair_off == 0)
                        elif isinstance(off_indicators[i+1], int):
                            if off_indicators[i+1] == 1:
                                self.model.Add(pair_off == off_indicators[i])
                            else:
                                self.model.Add(pair_off == 0)
                        else:
                            # Both are variables
                            self.model.Add(pair_off <= off_indicators[i])
                            self.model.Add(pair_off <= off_indicators[i+1])
                            self.model.Add(pair_off >= off_indicators[i] + off_indicators[i+1] - 1)
                        
                        pair_vars.append(pair_off)
                    
                    if pair_vars:
                        has_pair = self.model.NewBoolVar(f'has_pair_{emp}_week{week}')
                        self.model.AddMaxEquality(has_pair, pair_vars)
                        reward_penalty = self.model.NewIntVar(
                            -self.CONSECUTIVE_OFF_REWARD, 
                            self.CONSECUTIVE_OFF_PENALTY,
                            f'reward_consec_off_{emp}_week{week}'
                        )
                        self.model.Add(reward_penalty == -self.CONSECUTIVE_OFF_REWARD).OnlyEnforceIf(has_pair)
                        self.model.Add(reward_penalty == self.CONSECUTIVE_OFF_PENALTY).OnlyEnforceIf(has_pair.Not())
                        self.penalties.append(reward_penalty)

    def _add_one_shift_per_day_penalty(self):
        # Penalize working multiple shifts in one day
        for emp in self.all_employees:
            for day in self.active_days:
                day_str = day.isoformat()
                if day_str in self.request.closed_days or day_str in self.employee_details[emp]['days_off']:
                    continue
                
                # Check for multiple shifts per day
                double_shift_var = self.employee_double_shifts.get(emp, {}).get(day_str)
                if double_shift_var is not None:
                    self.penalties.append(double_shift_var * self.PENALTY_WEIGHTS['one_shift'])

    def _add_isolated_workday_penalty(self):
        # Penalize isolated work days
        for emp in self.all_employees:
            for i in range(1, len(self.days) - 1):
                prev_day = self.days[i-1].isoformat()
                curr_day = self.days[i].isoformat()
                next_day = self.days[i+1].isoformat()
                
                if (curr_day in self.request.closed_days or 
                    curr_day in self.employee_details[emp]['vacation_days'] or
                    curr_day in self.employee_details[emp]['time_off_days'] or
                    prev_day in self.request.closed_days or
                    next_day in self.request.closed_days):
                    continue
                
                prev_off = prev_day in self.employee_details[emp]['vacation_days'] or prev_day in self.employee_details[emp]['time_off_days']
                next_off = next_day in self.employee_details[emp]['vacation_days'] or next_day in self.employee_details[emp]['time_off_days']
                
                # Get working variables if they exist
                prev_work_var = self.employee_working_vars.get(emp, {}).get(prev_day)
                curr_work_var = self.employee_working_vars.get(emp, {}).get(curr_day)
                next_work_var = self.employee_working_vars.get(emp, {}).get(next_day)
                
                if prev_work_var is not None and curr_work_var is not None and next_work_var is not None:
                    # Create off indicators (1 = off, 0 = working)
                    prev_off_var = self.model.NewBoolVar(f'prev_off_{emp}_{curr_day}')
                    next_off_var = self.model.NewBoolVar(f'next_off_{emp}_{curr_day}')
                    
                    self.model.Add(prev_off_var == 1 - prev_work_var)
                    self.model.Add(next_off_var == 1 - next_work_var)
                    
                    # Create isolated work day indicator
                    isolated = self.model.NewBoolVar(f'isolated_{emp}_{curr_day}')
                    self.model.AddBoolAnd([prev_off_var, curr_work_var, next_off_var]).OnlyEnforceIf(isolated)
                    self.model.AddBoolOr([
                        prev_off_var.Not(), curr_work_var.Not(), next_off_var.Not()
                    ]).OnlyEnforceIf(isolated.Not())
                    
                    self.penalties.append(isolated * 300)  # Penalty for isolated work day


    def _add_consecutive_double_shift_constraint(self):
        # Penalize consecutive days with double shifts
        for emp in self.all_employees:
        # Iterate through all consecutive day pairs
            for i in range(len(self.days) - 1):
                day1 = self.days[i]
                day2 = self.days[i + 1]
                day1_str = day1.isoformat()
                day2_str = day2.isoformat()

                # Check if both days are active for the employee
                if (day1_str in self.employee_double_shifts.get(emp, {}) and 
                    day2_str in self.employee_double_shifts.get(emp, {})):
                    
                    # Get double shift variables
                    double1 = self.employee_double_shifts[emp][day1_str]
                    double2 = self.employee_double_shifts[emp][day2_str]

                    # Create penalty variable
                    penalty_var = self.model.NewBoolVar(f'penalty_{emp}_{day1_str}_{day2_str}')
                    self.model.AddBoolAnd([double1, double2]).OnlyEnforceIf(penalty_var)
                    self.model.AddBoolOr([double1.Not(), double2.Not()]).OnlyEnforceIf(penalty_var.Not())
                    
                    # Add weighted penalty
                    self.penalties.append(penalty_var * self.PENALTY_WEIGHTS['consecutive_double'])
                    

    def solve(self):
        # Create all variables
        self._create_assignment_variables()
        self._create_working_day_indicators()
        self._create_shift_indicators()
        self._create_double_shift_indicators()
        
        # Add all constraints
        self._add_task_coverage_constraint()
        self._add_one_task_per_employee_constraint()
        self._add_supervisor_constraint()
        self._add_max_consecutive_days_constraint()
        self._add_two_days_off_per_week_constraint()
        self._add_consecutive_double_shift_constraint()
        
        # Add soft constraints with penalties
        self._add_contract_hours_constraint()
        self._add_fairness_constraint()
        self._add_consecutive_off_days_reward()
        self._add_one_shift_per_day_penalty()
        self._add_isolated_workday_penalty()
        
        # Set objective: minimize total penalties
        if self.penalties:
            self.model.Minimize(sum(self.penalties))
        
        # Solve with parameters from colab.py
        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = 60  # 1 minute timeout
        solver.parameters.num_search_workers = 8   # Use 8 workers for parallel search
        
        status = solver.Solve(self.model)
        
        if status in [cp_model.OPTIMAL, cp_model.FEASIBLE]:
            return self._extract_solution(solver)
        else:
            raise ValueError(f"No solution found. Status: {solver.StatusName(status)}")

    def _extract_solution(self, solver: cp_model.CpSolver) -> dict:
        schedule = {}
        total_hours = {emp: 0 for emp in self.all_employees}
        daily_hours = {day.isoformat(): 0 for day in self.days}
        violations = []

        # Initialize schedule dictionary
        for emp in self.all_employees:
            schedule[emp] = {}
            emp_info = self.employee_details[emp]
            for day in self.days:
                day_str = day.isoformat()
                schedule[emp][day_str] = []
                
                if day_str in self.request.closed_days:
                    schedule[emp][day_str].append('CLOSED')
                else:
                    is_vacation = any(
                        req.get('type') == 'Vacation Leave' 
                        and req.get('status') == 'Approved'
                        and day_str in req.get('dates', [])
                        for req in emp_info.get('requests', [])
                    )
                    
                    if is_vacation:
                        schedule[emp][day_str] = ['VACATION']
                    elif day_str in emp_info['time_off_days']:
                        schedule[emp][day_str] = ['TIME OFF']
                    else:
                        for key, value in self.assignments.items():
                            e, d, shift_name, task_name = key
                            var, _ = value
                            if e == emp and d == day_str and solver.Value(var):
                                date_obj = datetime.strptime(day_str, "%Y-%m-%d")
                                day_of_week = date_obj.strftime('%A')
                                
                                task_config = next(
                                    (t for shift in self.request.shifts if shift['name'] == shift_name 
                                    for t in shift['tasks'] if t['name'] == task_name),
                                    None
                                )
                                
                                actual_duration = 0
                                time_info = {'startTime': '00:00', 'endTime': '00:00'}
                                
                                if task_config:
                                    task_schedule = task_config.get('schedule', [])
                                    day_schedule = next(
                                        (ts for ts in task_schedule if ts['day'] == day_of_week),
                                        {'startTime': '00:00', 'endTime': '00:00'}
                                    )
                                    time_info = day_schedule
                                    
                                    start = datetime.strptime(time_info.get('startTime', '00:00'), "%H:%M")
                                    end = datetime.strptime(time_info.get('endTime', '00:00'), "%H:%M")
                                    actual_duration = (end - start).seconds // 3600

                                total_hours[emp] += actual_duration
                                
                                task_entry = {
                                    'task': task_name,
                                    'shift': shift_name,
                                    'start': time_info.get('startTime', '00:00'),
                                    'end': time_info.get('endTime', '00:00'),
                                    'duration_hours': actual_duration
                                }

                                schedule[emp][day_str].append(task_entry)
                                daily_hours[day_str] += actual_duration

        # Check for actual violations in the solution
        for violation in self.violations:
            if violation.type == "SUPERVISOR":
                # Check if any supervisor was actually assigned
                for date in violation.dates:
                    shift = violation.details["shift"]
                    has_supervisor = False
                    supervisor_details = []
                    
                    # Get all supervisors and their skills
                    for emp in self.all_employees:
                        if self.employee_details[emp]['position'] == 'supervisor':
                            supervisor_details.append({
                                'name': emp,
                                'skills': list(self.employee_details[emp]['skills'])
                            })
                    
                    for emp in self.all_employees:
                        if (self.employee_details[emp]['position'] == 'supervisor' and
                            any(entry.get('shift') == shift for entry in schedule[emp][date])):
                            has_supervisor = True
                            break
                    
                    if not has_supervisor:
                        violation.details.update({
                            'available_supervisors': supervisor_details,
                            'shift_requirements': self.shift_structure[shift]['required_skills']
                        })
                        violations.append(violation.to_dict())

            elif violation.type == "CONSECUTIVE_DAYS":
                # Check actual consecutive working days
                emp = violation.employees[0]
                consecutive_count = 0
                consecutive_days = []
                
                for date in violation.dates:
                    if any(isinstance(entry, dict) for entry in schedule[emp][date]):
                        consecutive_count += 1
                        consecutive_days.append({
                            'date': date,
                            'shifts': [entry for entry in schedule[emp][date] if isinstance(entry, dict)]
                        })
                    else:
                        consecutive_count = 0
                        consecutive_days = []
                    
                    if consecutive_count > 5:
                        violation.details.update({
                            'consecutive_days': consecutive_days,
                            'total_days': consecutive_count
                        })
                        violations.append(violation.to_dict())
                        break

            elif violation.type == "CONTRACT_HOURS":
                # Check actual hours against contract
                emp = violation.employees[0]
                week_hours = 0
                daily_breakdown = []
                
                for date in violation.dates:
                    day_hours = sum(entry.get('duration_hours', 0) for entry in schedule[emp][date] if isinstance(entry, dict))
                    week_hours += day_hours
                    daily_breakdown.append({
                        'date': date,
                        'hours': day_hours,
                        'shifts': [entry for entry in schedule[emp][date] if isinstance(entry, dict)]
                    })
                
                if week_hours < violation.details["min_hours"] or week_hours > violation.details["max_hours"]:
                    violation.details.update({
                        'actual_hours': week_hours,
                        'daily_breakdown': daily_breakdown,
                        'contract': self.employee_details[emp]['contract']
                    })
                    violations.append(violation.to_dict())

        # Check fairness violations
        # if len(total_hours) >= 2:
        #     max_hours = max(total_hours.values())
        #     min_hours = min(total_hours.values())
        #     hours_diff = max_hours - min_hours
            
        #     if hours_diff > 20:  # Threshold for fairness violation
        #         violations.append({
        #             "type": "FAIRNESS",
        #             "message": "Significant difference in total hours between employees",
        #             "employees": list(total_hours.keys()),
        #             "dates": [d.isoformat() for d in self.days],
        #             "details": {
        #                 "max_hours": max_hours,
        #                 "min_hours": min_hours,
        #                 "hours_difference": hours_diff,
        #                 "employee_hours": total_hours
        #             }
        #         })

        # Check consecutive off days violations
        for emp in self.all_employees:
            for week in range(self.num_full_weeks):
                week_days = self.days[week*7:(week+1)*7]
                off_days = []
                
                for day in week_days:
                    day_str = day.isoformat()
                    if (day_str in self.request.closed_days or 
                        day_str in self.employee_details[emp]['days_off'] or
                        not any(isinstance(entry, dict) for entry in schedule[emp][day_str])):
                        off_days.append(day_str)
                
                if len(off_days) < 2:
                    violations.append({
                        "type": "TWO_DAYS_OFF",
                        "message": f"Employee {emp} has less than 2 days off in week starting {week_days[0].isoformat()}",
                        "employees": [emp],
                        "dates": week_days,
                        "details": {
                            "off_days": off_days,
                            "total_off_days": len(off_days),
                            "required_off_days": 2
                        }
                    })

        # Check double shift violations
        for emp in self.all_employees:
            for i in range(len(self.days) - 1):
                day1 = self.days[i]
                day2 = self.days[i + 1]
                day1_str = day1.isoformat()
                day2_str = day2.isoformat()
                
                if (day1_str in self.request.closed_days or 
                    day2_str in self.request.closed_days or
                    day1_str in self.employee_details[emp]['days_off'] or
                    day2_str in self.employee_details[emp]['days_off']):
                    continue
                
                day1_shifts = [entry for entry in schedule[emp][day1_str] if isinstance(entry, dict)]
                day2_shifts = [entry for entry in schedule[emp][day2_str] if isinstance(entry, dict)]
                
                if len(day1_shifts) > 1 and len(day2_shifts) > 1:
                    violations.append({
                        "type": "CONSECUTIVE_DOUBLE_SHIFTS",
                        "message": f"Employee {emp} has multiple shifts on consecutive days {day1_str} and {day2_str}",
                        "employees": [emp],
                        "dates": [day1_str, day2_str],
                        "details": {
                            "day1": {
                                "date": day1_str,
                                "shifts": day1_shifts,
                                "total_shifts": len(day1_shifts),
                                "total_hours": sum(shift.get('duration_hours', 0) for shift in day1_shifts)
                            },
                            "day2": {
                                "date": day2_str,
                                "shifts": day2_shifts,
                                "total_shifts": len(day2_shifts),
                                "total_hours": sum(shift.get('duration_hours', 0) for shift in day2_shifts)
                            }
                        }
                    })

        # Check isolated workday violations
        for emp in self.all_employees:
            for i in range(1, len(self.days) - 1):
                prev_day = self.days[i-1].isoformat()
                curr_day = self.days[i].isoformat()
                next_day = self.days[i+1].isoformat()
                
                if (curr_day in self.request.closed_days or 
                    curr_day in self.employee_details[emp]['days_off'] or
                    prev_day in self.request.closed_days or
                    next_day in self.request.closed_days):
                    continue
                
                curr_working = any(isinstance(entry, dict) for entry in schedule[emp][curr_day])
                prev_working = any(isinstance(entry, dict) for entry in schedule[emp][prev_day])
                next_working = any(isinstance(entry, dict) for entry in schedule[emp][next_day])
                
                if curr_working and not prev_working and not next_working:
                    violations.append({
                        "type": "ISOLATED_WORKDAY",
                        "message": f"Employee {emp} has an isolated workday on {curr_day}",
                        "employees": [emp],
                        "dates": [curr_day],
                        "details": {
                            "current_day": {
                                "date": curr_day,
                                "shifts": [entry for entry in schedule[emp][curr_day] if isinstance(entry, dict)]
                            },
                            "previous_day": {
                                "date": prev_day,
                                "shifts": [entry for entry in schedule[emp][prev_day] if isinstance(entry, dict)]
                            },
                            "next_day": {
                                "date": next_day,
                                "shifts": [entry for entry in schedule[emp][next_day] if isinstance(entry, dict)]
                            }
                        }
                    })

        # Add shift preference markers
        for emp in self.all_employees:
            emp_info = self.employee_details[emp]
            preferred_shifts = emp_info.get('preferred_shifts_by_date', {})
            for date_str, preferred_shift in preferred_shifts.items():
                if date_str in schedule[emp]:
                    worked_preferred = any(
                        entry.get('shift') == preferred_shift
                        for entry in schedule[emp][date_str]
                        if isinstance(entry, dict)
                    )
                    if worked_preferred:
                        schedule[emp][date_str].insert(0, f'PREFERRED: {preferred_shift}')
                    else:
                        schedule[emp][date_str].insert(0, 'MISSED PREFERENCE')
                        # Add violation for missed preference with detailed information
                        actual_shifts = [entry for entry in schedule[emp][date_str] if isinstance(entry, dict)]
                        violations.append({
                            "type": "PREFERENCE",
                            "message": f"Employee {emp} did not get preferred shift {preferred_shift}",
                            "employees": [emp],
                            "dates": [date_str],
                            "details": {
                                "preferred_shift": preferred_shift,
                                "assigned_shifts": actual_shifts,
                                "employee_position": emp_info['position'],
                                "employee_skills": list(emp_info['skills'])
                            }
                        })

        # Add 'OFF' for days with no entries
        for emp in self.all_employees:
            for day in self.days:
                day_str = day.isoformat()
                if not schedule[emp][day_str]:
                    if day_str in self.request.closed_days:
                        schedule[emp][day_str].append('CLOSED')
                    else:
                        schedule[emp][day_str].append('OFF')

        return {
            "schedule": schedule,
            "summary": {
                "total_man_hours": sum(daily_hours.values()),
                "daily_man_hours": daily_hours,
                "total_hours_per_employee": total_hours
            },
            "violations": violations
        }
    


@app.post("/api/generate-schedule")
async def generate_schedule(request: SolverRequest):
    try:
        for shift_idx, shift in enumerate(request.shifts):
            for task_idx, task in enumerate(shift.get('tasks', [])):
                if not task.get('skillRequirement') and not task.get('skill_requirement'):
                    raise HTTPException(
                        status_code=400,
                        detail=f"Missing skill requirement in shift {shift_idx+1} task {task_idx+1}"
                    )
        
        generator = ImprovedScheduleGenerator(request)
        return generator.solve()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

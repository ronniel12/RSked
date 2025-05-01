from fastapi import FastAPI, HTTPException, Request, Depends, WebSocket, WebSocketDisconnect
from pydantic import BaseModel, Field
from ortools.sat.python import cp_model
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Set
from pathlib import Path
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import math
import json
from sqlalchemy import create_engine, Column, Integer, String, JSON, DateTime, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session, joinedload

app = FastAPI()

from fastapi.encoders import jsonable_encoder

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Database setup
SQLALCHEMY_DATABASE_URL = "sqlite:///./schedules.db"
engine = create_engine(SQLALCHEMY_DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# Database models
class Schedule(Base):
    __tablename__ = "schedules"
    
    id = Column(Integer, primary_key=True, index=True)
    employee = Column(String, index=True)
    date = Column(String, index=True)
    shift_data = Column(JSON)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

class Employee(Base):
    __tablename__ = "employees"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, index=True)
    skills = Column(JSON)
    position = Column(String)
    contract_id = Column(Integer, ForeignKey('contracts.id'))
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

class Shift(Base):
    __tablename__ = "shifts"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, index=True)
    tasks = Column(JSON)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

class EmployeeRequest(Base):
    __tablename__ = "requests"
    
    id = Column(Integer, primary_key=True, index=True)
    employee_id = Column(Integer, ForeignKey('employees.id'))
    type = Column(String)
    dates = Column(JSON)
    details = Column(JSON)
    status = Column(String)
    shift_preference = Column(JSON)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

class Skill(Base):
    __tablename__ = "skills"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, unique=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

class Contract(Base):
    __tablename__ = "contracts"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, unique=True)
    min_hours = Column(Integer)
    max_hours = Column(Integer)
    unit_days = Column(Integer)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

class Setting(Base):
    __tablename__ = "settings"
    
    id = Column(Integer, primary_key=True, index=True)
    key = Column(String, unique=True)
    value = Column(JSON)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

# Create tables
Base.metadata.create_all(bind=engine)

# Dependency
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@app.get("/", response_class=HTMLResponse)
async def serve_frontend(request: Request):
    index_path = Path(__file__).parent / "static" / "index.html"
    with open(index_path, encoding='utf-8') as f:
        return HTMLResponse(content=f.read(), status_code=200)

# CRUD operations

@app.get("/api/schedules")
def get_schedules(db: Session = Depends(get_db)):
    schedules = db.query(Schedule).all()
    return jsonable_encoder(schedules)

from fastapi import Body

@app.post("/api/save-schedule")
async def save_generated_schedule(
    data: dict = Body(...),
    db: Session = Depends(get_db)
):
    # Save the entire generated schedule as a JSON blob
    try:
        schedule_blob = {
            "schedule": data.get("schedule"),
            "startDate": data.get("startDate"),
            "duration": data.get("duration"),
            "saved_at": datetime.utcnow().isoformat()
        }
        db_schedule = Schedule(
            employee="ALL",
            date=schedule_blob["startDate"] or datetime.utcnow().date().isoformat(),
            shift_data=schedule_blob
        )
        db.add(db_schedule)
        db.commit()
        db.refresh(db_schedule)
        return {"status": "success", "id": db_schedule.id}
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Failed to save schedule: {str(e)}")

@app.get("/api/schedules/{employee_id}")
async def get_schedule(employee_id: str, date: str, db: Session = Depends(get_db)):
    schedule = db.query(Schedule).filter(
        Schedule.employee == employee_id,
        Schedule.date == date
    ).first()
    if not schedule:
        raise HTTPException(status_code=404, detail="Schedule not found")
    return schedule.shift_data

@app.post("/api/schedules")
async def create_schedule(schedule_data: dict, db: Session = Depends(get_db)):
    db_schedule = Schedule(
        employee=schedule_data['employee'],
        date=schedule_data['date'],
        shift_data=schedule_data['shift']
    )
    db.add(db_schedule)
    db.commit()
    db.refresh(db_schedule)
    return {"status": "success", "id": db_schedule.id}

@app.put("/api/schedules/{schedule_id}")
async def update_schedule(schedule_id: int, update_data: dict, db: Session = Depends(get_db)):
    db_schedule = db.query(Schedule).filter(Schedule.id == schedule_id).first()
    if not db_schedule:
        raise HTTPException(status_code=404, detail="Schedule not found")
    
    # Update fields
    for field, value in update_data.items():
        if field == 'employee':
            db_schedule.employee = value
        elif field == 'date':
            db_schedule.date = value
        elif field == 'shift':
            db_schedule.shift_data = value
    
    db.commit()
    return {"status": "success", "message": "Schedule updated"}

@app.delete("/api/schedules/{schedule_id}")
async def delete_schedule(schedule_id: int, db: Session = Depends(get_db)):
    db_schedule = db.query(Schedule).filter(Schedule.id == schedule_id).first()
    if not db_schedule:
        raise HTTPException(status_code=404, detail="Schedule not found")
    
    db.delete(db_schedule)
    db.commit()
    return {"status": "success", "message": "Schedule deleted"}

# Update the existing update endpoint to use the CRUD operations
@app.put("/api/update-schedule")
async def update_schedule(update_data: dict, db: Session = Depends(get_db)):
    try:
        employee = update_data['employee']
        date = update_data['date']
        new_shift = update_data['shift']
        
        if not employee or not date:
            raise ValueError("Employee and date are required fields")
        
        # Validate shift data
        if isinstance(new_shift, list):
            for shift in new_shift:
                if isinstance(shift, dict) and 'shift' in shift and 'task' in shift:
                    if 'start' not in shift or 'end' not in shift:
                        raise ValueError(f"Start and end times are required for shift {shift.get('shift')}")
                    
                    if 'duration_hours' not in shift:
                        try:
                            start_time = datetime.strptime(shift['start'], "%H:%M").time()
                            end_time = datetime.strptime(shift['end'], "%H:%M").time()
                            duration = (datetime.combine(datetime.min, end_time) - 
                                      datetime.combine(datetime.min, start_time)).seconds // 3600
                            shift['duration_hours'] = duration
                        except ValueError:
                            raise ValueError(f"Invalid time format in shift {shift.get('shift')}")
        
        # Check if schedule exists
        existing_schedule = db.query(Schedule).filter(
            Schedule.employee == employee,
            Schedule.date == date
        ).first()
        
        if existing_schedule:
            # Update existing
            existing_schedule.shift_data = new_shift
            db.commit()
            return {
                "status": "success", 
                "message": "Schedule updated successfully",
                "employee": employee,
                "date": date,
                "updated_shift": new_shift
            }
        else:
            # Create new
            db_schedule = Schedule(
                employee=employee,
                date=date,
                shift_data=new_shift
            )
            db.add(db_schedule)
            db.commit()
            return {
                "status": "success", 
                "message": "Schedule created successfully",
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
        self.employee_details = {}  # Initialize the dictionary
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

        # Initialize employee details for all employees first
        for emp in self.request.employees:
            skills = emp.get('skills', [])
            if isinstance(skills, str):
                skills = [skills]
            
            self.all_employees.append(emp['name'])
            self.employee_details[emp['name']] = {
                'skills': set(skills),
                'position': emp.get('position', 'Employee').lower(),
                'contract': emp.get('contract', {}),
                'vacation_days': set(),
                'time_off_days': set(),
                'available_days': [],
                'preferred_shifts_by_date': {},
                'days_off': set(),
                'requests': emp.get('requests', [])
            }

        # Get database session
        db = SessionLocal()
        try:
            for emp in self.request.employees:
                emp_name = emp['name']
                emp_details = self.employee_details[emp_name]
                
                # Look up employee in database to get ID
                db_employee = db.query(Employee).filter(Employee.name == emp_name).first()
                if not db_employee:
                    raise ValueError(f"Employee {emp_name} not found in database")
                
                # Fetch requests from database
                db_requests = db.query(EmployeeRequest).filter(
                    EmployeeRequest.employee_id == db_employee.id,
                    EmployeeRequest.status == 'Approved'
                ).all()
                
                for request in db_requests:
                    if request.status == 'Approved':
                        if request.type == 'Vacation Leave':
                            for date_str in request.dates:
                                try:
                                    date = datetime.strptime(date_str, "%Y-%m-%d").date()
                                    if self.start_date <= date < (self.start_date + timedelta(days=self.request.schedule_days)):
                                        emp_details['vacation_days'].add(date.isoformat())
                                except ValueError:
                                    continue
                        elif request.type == 'Time Off':
                            for date_str in request.dates:
                                try:
                                    date = datetime.strptime(date_str, "%Y-%m-%d").date()
                                    if self.start_date <= date < (self.start_date + timedelta(days=self.request.schedule_days)):
                                        emp_details['time_off_days'].add(date.isoformat())
                                except ValueError:
                                    continue

                # Update available days based on vacation and time off
                emp_details['available_days'] = [
                    day.isoformat() for day in self.days 
                    if day.isoformat() not in self.request.closed_days 
                    and day.isoformat() not in emp_details['vacation_days']
                    and day.isoformat() not in emp_details['time_off_days']
                ]
                
                # Update days_off
                emp_details['days_off'] = emp_details['vacation_days'].union(emp_details['time_off_days'])

                # Handle shift preferences
                for request in emp.get('requests', []):
                    if request.get('type') == 'Shift Preference' and request.get('status') == 'Approved':
                        for date_str in request.get('dates', []):
                            emp_details['preferred_shifts_by_date'][date_str] = request.get('shiftPreference')

        except Exception as e:
            raise ValueError(f"Error preparing data structures: {str(e)}")
        finally:
            db.close()

        # Process shifts and tasks
        for shift in self.request.shifts:
            shift_name = shift['name']
            tasks_info = []
            for task in shift.get('tasks', []):
                task_schedule = {}
                for day_sched in task.get('schedule', []):
                    try:
                        start_time_str = day_sched.get('startTime', '00:00')
                        end_time_str = day_sched.get('endTime', '00:00')
                        
                        # Skip if times are '00:00' or '--:--'
                        if start_time_str in ('00:00', '--:--') or end_time_str in ('00:00', '--:--'):
                            continue
                            
                        start = datetime.strptime(start_time_str, "%H:%M").time()
                        end = datetime.strptime(end_time_str, "%H:%M").time()
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
        for emp in self.all_employees:
            emp_info = self.employee_details[emp]
            contract = emp_info['contract']
            
            # Get contract parameters
            min_hours = int(contract.get('minHours', 0))
            max_hours = int(contract.get('maxHours', 168))
            unit_days = int(contract.get('unitDays', 7))
            
            # Calculate number of complete units in the schedule period
            total_days = len(self.days)
            num_complete_units = total_days // unit_days
            remaining_days = total_days % unit_days
            
            # Calculate working days in each unit period
            for unit in range(num_complete_units):
                start_idx = unit * unit_days
                end_idx = start_idx + unit_days
                unit_days_range = self.days[start_idx:end_idx]
                
                # Calculate actual working days in this unit (excluding closed days and time off)
                working_days_in_unit = sum(1 for day in unit_days_range 
                    if day.isoformat() not in self.request.closed_days 
                    and day.isoformat() not in emp_info['days_off'])
                
                # Adjust min/max hours proportionally if there are non-working days
                if working_days_in_unit < unit_days:
                    adjusted_min = math.floor(min_hours * (working_days_in_unit / unit_days))
                    adjusted_max = math.ceil(max_hours * (working_days_in_unit / unit_days))
                else:
                    adjusted_min = min_hours
                    adjusted_max = max_hours
                
                # Create variable for total hours in this unit
                unit_hours = self.model.NewIntVar(0, 168, f'unit_hours_{emp}_{unit}')
                
                # Sum up all assignments for this employee in this unit period
                unit_assignments = []
                for day in unit_days_range:
                    day_str = day.isoformat()
                    if day_str not in self.request.closed_days and day_str not in emp_info['days_off']:
                        for key, (var, duration) in self.assignments.items():
                            e, d, _, _ = key
                            if e == emp and d == day_str:
                                unit_assignments.append(var * duration)
                
                if unit_assignments:
                    self.model.Add(unit_hours == sum(unit_assignments))
                    
                    # Add soft constraints with penalties for this unit
                    under = self.model.NewIntVar(0, adjusted_max, f'under_{emp}_{unit}')
                    over = self.model.NewIntVar(0, adjusted_max, f'over_{emp}_{unit}')
                    
                    self.model.Add(under >= adjusted_min - unit_hours)
                    self.model.Add(over >= unit_hours - adjusted_max)
                    
                    self.penalties.append(under * self.PENALTY_WEIGHTS['hours'])
                    self.penalties.append(over * self.PENALTY_WEIGHTS['hours'])
                    
                    # Store unit information for violation checking
                    if not hasattr(self, 'unit_periods'):
                        self.unit_periods = {}
                    if emp not in self.unit_periods:
                        self.unit_periods[emp] = []
                    
                    self.unit_periods[emp].append({
                        'unit': unit,
                        'start_date': unit_days_range[0].isoformat(),
                        'end_date': unit_days_range[-1].isoformat(),
                        'min_hours': adjusted_min,
                        'max_hours': adjusted_max,
                        'working_days': working_days_in_unit,
                        'hours_var': unit_hours
                    })
            
            # Handle remaining days if any
            if remaining_days > 0:
                start_idx = num_complete_units * unit_days
                remaining_range = self.days[start_idx:]
                
                working_days_remaining = sum(1 for day in remaining_range 
                    if day.isoformat() not in self.request.closed_days 
                    and day.isoformat() not in emp_info['days_off'])
                
                if working_days_remaining > 0:
                    # Adjust hours proportionally for remaining days
                    adjusted_min = math.floor(min_hours * (working_days_remaining / unit_days))
                    adjusted_max = math.ceil(max_hours * (working_days_remaining / unit_days))
                    
                    remaining_hours = self.model.NewIntVar(0, 168, f'remaining_hours_{emp}')
                    
                    remaining_assignments = []
                    for day in remaining_range:
                        day_str = day.isoformat()
                        if day_str not in self.request.closed_days and day_str not in emp_info['days_off']:
                            for key, (var, duration) in self.assignments.items():
                                e, d, _, _ = key
                                if e == emp and d == day_str:
                                    remaining_assignments.append(var * duration)
                    
                    if remaining_assignments:
                        self.model.Add(remaining_hours == sum(remaining_assignments))
                        
                        under = self.model.NewIntVar(0, adjusted_max, f'under_{emp}_remaining')
                        over = self.model.NewIntVar(0, adjusted_max, f'over_{emp}_remaining')
                        
                        self.model.Add(under >= adjusted_min - remaining_hours)
                        self.model.Add(over >= remaining_hours - adjusted_max)
                        
                        self.penalties.append(under * self.PENALTY_WEIGHTS['hours'])
                        self.penalties.append(over * self.PENALTY_WEIGHTS['hours'])
                        
                        # Store remaining period information
                        if emp in self.unit_periods:
                            self.unit_periods[emp].append({
                                'unit': 'remaining',
                                'start_date': remaining_range[0].isoformat(),
                                'end_date': remaining_range[-1].isoformat(),
                                'min_hours': adjusted_min,
                                'max_hours': adjusted_max,
                                'working_days': working_days_remaining,
                                'hours_var': remaining_hours
                            })

    def _check_contract_violations(self, schedule, solver):
        violations = []
        
        for emp in self.all_employees:
            if not hasattr(self, 'unit_periods') or emp not in self.unit_periods:
                continue
                
            emp_info = self.employee_details[emp]
            contract = emp_info['contract']
            
            for period in self.unit_periods[emp]:
                actual_hours = solver.Value(period['hours_var'])
                
                if actual_hours < period['min_hours'] or actual_hours > period['max_hours']:
                    # Calculate daily breakdown for this period
                    daily_breakdown = []
                    current_date = datetime.strptime(period['start_date'], "%Y-%m-%d").date()
                    end_date = datetime.strptime(period['end_date'], "%Y-%m-%d").date()
                    
                    while current_date <= end_date:
                        date_str = current_date.isoformat()
                        shifts = schedule[emp][date_str]
                        day_hours = sum(
                            shift['duration_hours'] 
                            for shift in shifts 
                            if isinstance(shift, dict)
                        )
                        
                        if day_hours > 0:
                            daily_breakdown.append({
                                'date': date_str,
                                'hours': day_hours,
                                'shifts': [s for s in shifts if isinstance(s, dict)]
                            })
                        
                        current_date += timedelta(days=1)
                    
                    violations.append({
                        "type": "CONTRACT_HOURS",
                        "message": f"Employee {emp} worked {actual_hours} hours in period {period['start_date']} to {period['end_date']} (allowed range: {period['min_hours']}-{period['max_hours']})",
                        "employees": [emp],
                        "dates": [period['start_date'], period['end_date']],
                        "details": {
                            "period_start": period['start_date'],
                            "period_end": period['end_date'],
                            "actual_hours": actual_hours,
                            "min_hours": period['min_hours'],
                            "max_hours": period['max_hours'],
                            "working_days": period['working_days'],
                            "daily_breakdown": daily_breakdown,
                            "contract": {
                                "name": contract.get('name', 'Unknown'),
                                "min_hours": contract.get('minHours'),
                                "max_hours": contract.get('maxHours'),
                                "unit_days": contract.get('unitDays')
                            }
                        }
                    })
        
        return violations

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
            
            # Get all approved requests for this employee
            vacation_days = emp_info['vacation_days']
            time_off_days = emp_info['time_off_days']
            preferred_shifts = emp_info.get('preferred_shifts_by_date', {})
            
            for day in self.days:
                day_str = day.isoformat()
                schedule[emp][day_str] = []
                
                # First check for closed days
                if day_str in self.request.closed_days:
                    schedule[emp][day_str] = ['CLOSED']
                    continue
                
                # Then check for vacation days
                if day_str in vacation_days:
                    schedule[emp][day_str] = ['VACATION']
                    continue
                
                # Then check for time off
                if day_str in time_off_days:
                    schedule[emp][day_str] = ['TIME OFF']
                    continue
                
                # Get all assignments for this day
                day_assignments = []
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
                        day_assignments.append(task_entry)
                        daily_hours[day_str] += actual_duration
                
                # If we have assignments for this day
                if day_assignments:
                    # Check for shift preferences first
                    if day_str in preferred_shifts:
                        preferred_shift = preferred_shifts[day_str]
                        if any(assignment['shift'] == preferred_shift for assignment in day_assignments):
                            schedule[emp][day_str].append(f'PREFERRED: {preferred_shift}')
                        else:
                            schedule[emp][day_str].append('MISSED PREFERENCE')
                    
                    # Add all assignments
                    schedule[emp][day_str].extend(day_assignments)
                else:
                    # If no assignments and not a special day, mark as OFF
                    schedule[emp][day_str] = ['OFF']

        # Check for contract violations
        contract_violations = self._check_contract_violations(schedule, solver)
        violations.extend(contract_violations)
        
        # Check for other violations...
        # [Rest of the violation checking code remains the same]

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
        print("Received request:", request)
        
        for shift_idx, shift in enumerate(request.shifts):
            for task_idx, task in enumerate(shift.get('tasks', [])):
                if not task.get('skillRequirement') and not task.get('skill_requirement'):
                    raise HTTPException(
                        status_code=400,
                        detail=f"Missing skill requirement in shift {shift_idx+1} task {task_idx+1}"
                    )
        
        generator = ImprovedScheduleGenerator(request)
        result = generator.solve()
        # Detailed violation logging
        violations = result.get('violations', [])
        print("\n=== CONSTRAINT VIOLATION REPORT ===")
        for violation in violations:
            print(f"\nType: {violation['type']}")
            print(f"Message: {violation['message']}")
            print(f"Employees: {', '.join(violation['employees'])}")
            print(f"Dates: {', '.join(violation['dates'])}")
            print("Details:")
            for key, value in violation['details'].items():
                print(f"  {key}: {value}")
        print("\n=== END OF VIOLATION REPORT ===\n")
        
        return result
    except Exception as e:
        print("Error generating schedule:", str(e))
        raise HTTPException(status_code=500, detail=str(e))

class ScheduleEditRequest(BaseModel):
    employee_from: str
    employee_to: str
    date: str
    task_from: str
    task_to: str
    shift_from: str
    shift_to: str

class ScheduleEditResponse(BaseModel):
    status: str
    message: str
    new_schedule: Optional[Dict[str, Any]] = None
    violations: Optional[List[Dict[str, Any]]] = None
    can_undo: bool = False

@app.post("/api/edit-schedule", response_model=ScheduleEditResponse)
async def edit_schedule(edit_data: ScheduleEditRequest, db: Session = Depends(get_db)):
    try:
        # Get both employee schedules for the date
        schedule_from = db.query(Schedule).filter(
            Schedule.employee == edit_data.employee_from,
            Schedule.date == edit_data.date
        ).first()
        
        schedule_to = db.query(Schedule).filter(
            Schedule.employee == edit_data.employee_to,
            Schedule.date == edit_data.date
        ).first()

        if not schedule_from or not schedule_to:
            raise HTTPException(
                status_code=404,
                detail="One or both schedules not found"
            )

        # Find the tasks to swap
        task_from = next(
            (t for t in schedule_from.shift_data 
             if isinstance(t, dict) and 
             t.get('task') == edit_data.task_from and 
             t.get('shift') == edit_data.shift_from),
            None
        )
        task_to = next(
            (t for t in schedule_to.shift_data 
             if isinstance(t, dict) and 
             t.get('task') == edit_data.task_to and 
             t.get('shift') == edit_data.shift_to),
            None
        )

        if not task_from or not task_to:
            raise HTTPException(
                status_code=404,
                detail="One or both tasks not found"
            )

        # Get employee details for skill validation
        employee_from_details = db.query(Employee).filter(
            Employee.name == edit_data.employee_from
        ).first()
        employee_to_details = db.query(Employee).filter(
            Employee.name == edit_data.employee_to
        ).first()

        if not employee_from_details or not employee_to_details:
            raise HTTPException(
                status_code=404,
                detail="Employee details not found"
            )

        # Validate skills
        if (task_to['skill_requirement'] not in employee_from_details.skills or
            task_from['skill_requirement'] not in employee_to_details.skills):
            raise HTTPException(
                status_code=400,
                detail="Employees don't have required skills for the tasks"
            )

        # Create backup for undo functionality
        backup = {
            edit_data.employee_from: schedule_from.shift_data.copy(),
            edit_data.employee_to: schedule_to.shift_data.copy()
        }

        # Create a temporary solver request for just this day
        solver_request = SolverRequest(
            employees=[
                {
                    "name": edit_data.employee_from,
                    "skills": employee_from_details.skills,
                    "position": employee_from_details.position,
                    "contract": employee_from_details.contract,
                    "requests": employee_from_details.requests
                },
                {
                    "name": edit_data.employee_to,
                    "skills": employee_to_details.skills,
                    "position": employee_to_details.position,
                    "contract": employee_to_details.contract,
                    "requests": employee_to_details.requests
                }
            ],
            shifts=[
                {
                    "name": edit_data.shift_from,
                    "tasks": [
                        {
                            "name": edit_data.task_from,
                            "skillRequirement": task_from['skill_requirement'],
                            "schedule": [{
                                "day": datetime.strptime(edit_data.date, "%Y-%m-%d").strftime('%A'),
                                "startTime": task_from['start'],
                                "endTime": task_from['end']
                            }]
                        }
                    ]
                },
                {
                    "name": edit_data.shift_to,
                    "tasks": [
                        {
                            "name": edit_data.task_to,
                            "skillRequirement": task_to['skill_requirement'],
                            "schedule": [{
                                "day": datetime.strptime(edit_data.date, "%Y-%m-%d").strftime('%A'),
                                "startTime": task_to['start'],
                                "endTime": task_to['end']
                            }]
                        }
                    ]
                }
            ],
            closed_days=[],
            start_date=edit_data.date,
            schedule_days=1
        )

        # Solve just for this day
        generator = ImprovedScheduleGenerator(solver_request)
        solution = generator.solve()

        # Check for violations
        if solution.get('violations'):
            return ScheduleEditResponse(
                status="warning",
                message="Swap would violate constraints",
                violations=solution['violations'],
                can_undo=False
            )

        # Update the schedules in the database
        schedule_from.shift_data = [
            t for t in schedule_from.shift_data 
            if not (isinstance(t, dict) and 
                   t.get('task') == edit_data.task_from and 
                   t.get('shift') == edit_data.shift_from)
        ]
        
        schedule_to.shift_data = [
            t for t in schedule_to.shift_data 
            if not (isinstance(t, dict) and 
                   t.get('task') == edit_data.task_to and 
                   t.get('shift') == edit_data.shift_to)
        ]

        # Add the new assignments from the solution
        for emp, assignments in solution['schedule'].items():
            db_schedule = db.query(Schedule).filter(
                Schedule.employee == emp,
                Schedule.date == edit_data.date
            ).first()
            
            if db_schedule:
                db_schedule.shift_data = assignments[edit_data.date]
                db.commit()

        # Store the backup in a temporary table for undo
        db_backup = ScheduleEditBackup(
            employee_from=edit_data.employee_from,
            employee_to=edit_data.employee_to,
            date=edit_data.date,
            backup_data=backup,
            expires_at=datetime.utcnow() + timedelta(minutes=30)  # Keep for 30 minutes
        )
        db.add(db_backup)
        db.commit()

        return ScheduleEditResponse(
            status="success",
            message="Schedule updated successfully",
            new_schedule=solution['schedule'],
            can_undo=True
        )

    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Unexpected error: {str(e)}"
        )

class ScheduleEditBackup(Base):
    __tablename__ = "schedule_edit_backups"
    
    id = Column(Integer, primary_key=True, index=True)
    employee_from = Column(String, index=True)
    employee_to = Column(String, index=True)
    date = Column(String, index=True)
    backup_data = Column(JSON)
    expires_at = Column(DateTime)
    created_at = Column(DateTime, default=datetime.utcnow)

@app.post("/api/undo-schedule-edit", response_model=ScheduleEditResponse)
async def undo_schedule_edit(edit_data: ScheduleEditRequest, db: Session = Depends(get_db)):
    try:
        # Get the most recent backup
        backup = db.query(ScheduleEditBackup).filter(
            ScheduleEditBackup.employee_from == edit_data.employee_from,
            ScheduleEditBackup.employee_to == edit_data.employee_to,
            ScheduleEditBackup.date == edit_data.date,
            ScheduleEditBackup.expires_at > datetime.utcnow()
        ).order_by(ScheduleEditBackup.created_at.desc()).first()

        if not backup:
            raise HTTPException(
                status_code=404,
                detail="No undo information available"
            )

        # Restore the schedules
        schedule_from = db.query(Schedule).filter(
            Schedule.employee == edit_data.employee_from,
            Schedule.date == edit_data.date
        ).first()
        
        schedule_to = db.query(Schedule).filter(
            Schedule.employee == edit_data.employee_to,
            Schedule.date == edit_data.date
        ).first()

        if schedule_from and schedule_to:
            schedule_from.shift_data = backup.backup_data[edit_data.employee_from]
            schedule_to.shift_data = backup.backup_data[edit_data.employee_to]
            db.commit()

            # Delete the backup
            db.delete(backup)
            db.commit()

            return ScheduleEditResponse(
                status="success",
                message="Schedule changes undone",
                new_schedule={
                    edit_data.employee_from: schedule_from.shift_data,
                    edit_data.employee_to: schedule_to.shift_data
                },
                can_undo=False
            )
        else:
            raise HTTPException(
                status_code=404,
                detail="Schedules not found for undo"
            )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error undoing schedule edit: {str(e)}"
        )

@app.post("/api/get-swap-options")
async def get_swap_options(swap_request: dict, db: Session = Depends(get_db)):
    """Get valid swap targets for drag-and-drop"""
    try:
        source_employee = swap_request['employee']
        source_date = swap_request['date']
        source_shift = swap_request['shift']
        source_task = swap_request['task']

        # Get all employees who could potentially swap
        potential_targets = []
        employees = db.query(Schedule).filter(Schedule.date == source_date).all()
        
        for emp_schedule in employees:
            if emp_schedule.employee == source_employee:
                continue
                
            # Check if any task in this schedule could be swapped
            for assignment in emp_schedule.shift_data:
                if isinstance(assignment, dict):
                    potential_targets.append({
                        'employee': emp_schedule.employee,
                        'date': source_date,
                        'shift': assignment.get('shift'),
                        'task': assignment.get('task')
                    })

        return {
            "status": "success",
            "options": potential_targets
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/process-swap")
async def process_swap(swap_data: dict, db: Session = Depends(get_db)):
    """Handle the actual swap operation"""
    try:
        source = swap_data['source']
        target = swap_data['target']
        
        # Get both schedules
        source_schedule = db.query(Schedule).filter(
            Schedule.employee == source['employee'],
            Schedule.date == source['date']
        ).first()
        
        target_schedule = db.query(Schedule).filter(
            Schedule.employee == target['employee'],
            Schedule.date == target['date']
        ).first()

        if not source_schedule or not target_schedule:
            raise HTTPException(status_code=404, detail="One or both schedules not found")

        # Find the specific tasks to swap
        source_task = next(
            (t for t in source_schedule.shift_data 
             if isinstance(t, dict) and 
             t.get('task') == source['task'] and 
             t.get('shift') == source['shift']),
            None
        )
        
        target_task = next(
            (t for t in target_schedule.shift_data 
             if isinstance(t, dict) and 
             t.get('task') == target['task'] and 
             t.get('shift') == target['shift']),
            None
        )

        if not source_task or not target_task:
            raise HTTPException(status_code=404, detail="One or both tasks not found")

        # Create backup for undo
        backup = {
            'source': source_schedule.shift_data.copy(),
            'target': target_schedule.shift_data.copy()
        }

        # Perform the swap
        source_schedule.shift_data = [
            t for t in source_schedule.shift_data 
            if not (isinstance(t, dict) and 
                   t.get('task') == source['task'] and 
                   t.get('shift') == source['shift'])
        ]
        
        target_schedule.shift_data = [
            t for t in target_schedule.shift_data 
            if not (isinstance(t, dict) and 
                   t.get('task') == target['task'] and 
                   t.get('shift') == target['shift'])
        ]

        # Add swapped tasks
        source_schedule.shift_data.append(target_task)
        target_schedule.shift_data.append(source_task)
        
        db.commit()

        return {
            "status": "success",
            "message": "Swap completed successfully",
            "new_schedules": {
                source['employee']: source_schedule.shift_data,
                target['employee']: target_schedule.shift_data
            }
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Employee CRUD
@app.get("/api/employees")
async def get_employees(db: Session = Depends(get_db)):
    return db.query(Employee).all()

@app.post("/api/employees")
async def create_employee(employee_data: dict, db: Session = Depends(get_db)):
    db_employee = Employee(**employee_data)
    db.add(db_employee)
    db.commit()
    db.refresh(db_employee)
    return db_employee

@app.put("/api/employees/{employee_id}")
async def update_employee(employee_id: int, employee_data: dict, db: Session = Depends(get_db)):
    db_employee = db.query(Employee).filter(Employee.id == employee_id).first()
    if not db_employee:
        raise HTTPException(status_code=404, detail="Employee not found")
    
    for key, value in employee_data.items():
        setattr(db_employee, key, value)
    
    db.commit()
    return db_employee

@app.delete("/api/employees/{employee_id}")
async def delete_employee(employee_id: int, db: Session = Depends(get_db)):
    db_employee = db.query(Employee).filter(Employee.id == employee_id).first()
    if not db_employee:
        raise HTTPException(status_code=404, detail="Employee not found")
    
    db.delete(db_employee)
    db.commit()
    return {"status": "success"}

# Shift CRUD
@app.get("/api/shifts")
async def get_shifts(db: Session = Depends(get_db)):
    return db.query(Shift).all()

@app.post("/api/shifts")
async def create_shift(shift_data: dict, db: Session = Depends(get_db)):
    db_shift = Shift(**shift_data)
    db.add(db_shift)
    db.commit()
    db.refresh(db_shift)
    return db_shift

@app.put("/api/shifts/{shift_id}")
async def update_shift(shift_id: int, shift_data: dict, db: Session = Depends(get_db)):
    db_shift = db.query(Shift).filter(Shift.id == shift_id).first()
    if not db_shift:
        raise HTTPException(status_code=404, detail="Shift not found")
    
    # Convert string timestamps to datetime objects if they exist
    if 'created_at' in shift_data and isinstance(shift_data['created_at'], str):
        shift_data['created_at'] = datetime.fromisoformat(shift_data['created_at'])
    if 'updated_at' in shift_data and isinstance(shift_data['updated_at'], str):
        shift_data['updated_at'] = datetime.fromisoformat(shift_data['updated_at'])
    
    for key, value in shift_data.items():
        setattr(db_shift, key, value)
    
    db.commit()
    return db_shift

@app.delete("/api/shifts/{shift_id}")
async def delete_shift(shift_id: int, db: Session = Depends(get_db)):
    db_shift = db.query(Shift).filter(Shift.id == shift_id).first()
    if not db_shift:
        raise HTTPException(status_code=404, detail="Shift not found")
    
    db.delete(db_shift)
    db.commit()
    return {"status": "success"}

# WebSocket connection manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def broadcast(self, message: str):
        for connection in self.active_connections:
            await connection.send_text(message)

manager = ConnectionManager()

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            # Keep the connection alive
            data = await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(websocket)

# Request CRUD
@app.get("/api/employee-requests")
async def get_employee_requests(db: Session = Depends(get_db)):
    requests = db.query(EmployeeRequest).all()
    # Transform the requests to include employee name
    result = []
    for req in requests:
        employee = db.query(Employee).filter(Employee.id == req.employee_id).first()
        employee_name = employee.name if employee else "Unknown"
        
        request_dict = {
            "id": req.id,
            "employee_id": req.employee_id,
            "employee": employee_name,
            "type": req.type,
            "dates": req.dates,
            "details": req.details,
            "status": req.status,
            "shiftPreference": req.shift_preference,
            "created_at": req.created_at.isoformat() if req.created_at else None,
            "updated_at": req.updated_at.isoformat() if req.updated_at else None
        }
        result.append(request_dict)
    
    return result

@app.post("/api/employee-requests")
async def create_employee_request(request_data: dict, db: Session = Depends(get_db)):
    # Look up employee ID by name
    employee = db.query(Employee).filter(Employee.name == request_data.get('employee')).first()
    if not employee:
        raise HTTPException(status_code=404, detail="Employee not found")
    
    # Create new request data with mapped fields
    request_dict = {
        'employee_id': employee.id,
        'type': request_data.get('type'),
        'dates': request_data.get('dates'),
        'details': request_data.get('details'),
        'status': request_data.get('status', 'Pending'),
        'shift_preference': request_data.get('shiftPreference')
    }
    
    db_request = EmployeeRequest(**request_dict)
    db.add(db_request)
    db.commit()
    db.refresh(db_request)
    
    # Convert to dict for the response
    response_data = {
        "id": db_request.id,
        "employee_id": db_request.employee_id,
        "employee": employee.name,
        "type": db_request.type,
        "dates": db_request.dates,
        "details": db_request.details,
        "status": db_request.status,
        "shiftPreference": db_request.shift_preference,
        "created_at": db_request.created_at.isoformat() if db_request.created_at else None,
        "updated_at": db_request.updated_at.isoformat() if db_request.updated_at else None
    }
    
    # Broadcast the new request
    await manager.broadcast(json.dumps({
        "action": "create",
        "type": "employee-request",
        "data": response_data
    }))
    
    return response_data

@app.put("/api/employee-requests/{request_id}")
async def update_employee_request(request_id: int, request_data: dict, db: Session = Depends(get_db)):
    db_request = db.query(EmployeeRequest).filter(EmployeeRequest.id == request_id).first()
    if not db_request:
        raise HTTPException(status_code=404, detail="Request not found")
    
    # Update fields based on mapping
    if 'employee' in request_data:
        employee = db.query(Employee).filter(Employee.name == request_data.get('employee')).first()
        if not employee:
            raise HTTPException(status_code=404, detail="Employee not found")
        db_request.employee_id = employee.id
    
    for field, value in request_data.items():
        if field == 'employee':
            continue  # Already handled above
        elif field == 'shiftPreference':
            db_request.shift_preference = value
        elif field in ['type', 'dates', 'details', 'status']:
            setattr(db_request, field, value)
    
    db.commit()
    db.refresh(db_request)
    
    # Get employee name for response
    employee = db.query(Employee).filter(Employee.id == db_request.employee_id).first()
    employee_name = employee.name if employee else "Unknown"
    
    # Prepare response data
    response_data = {
        "id": db_request.id,
        "employee_id": db_request.employee_id,
        "employee": employee_name,
        "type": db_request.type,
        "dates": db_request.dates,
        "details": db_request.details,
        "status": db_request.status,
        "shiftPreference": db_request.shift_preference,
        "created_at": db_request.created_at.isoformat() if db_request.created_at else None,
        "updated_at": db_request.updated_at.isoformat() if db_request.updated_at else None
    }
    
    # Broadcast the update
    await manager.broadcast(json.dumps({
        "action": "update",
        "type": "employee-request",
        "data": response_data
    }))
    
    return response_data

@app.delete("/api/employee-requests/{request_id}")
async def delete_employee_request(request_id: int, db: Session = Depends(get_db)):
    db_request = db.query(EmployeeRequest).filter(EmployeeRequest.id == request_id).first()
    if not db_request:
        raise HTTPException(status_code=404, detail="Request not found")
    
    # Store the ID before deletion for broadcasting
    deleted_id = db_request.id
    
    db.delete(db_request)
    db.commit()
    
    # Broadcast the deletion
    await manager.broadcast(json.dumps({
        "action": "delete",
        "type": "employee-request",
        "data": {"id": deleted_id}
    }))
    
    return {"status": "success", "message": "Request deleted"}

# Skill CRUD
@app.get("/api/skills")
async def get_skills(db: Session = Depends(get_db)):
    return db.query(Skill).all()

@app.post("/api/skills")
async def create_skill(skill_data: dict, db: Session = Depends(get_db)):
    db_skill = Skill(**skill_data)
    db.add(db_skill)
    db.commit()
    db.refresh(db_skill)
    return db_skill

@app.put("/api/skills/{skill_id}")
async def update_skill(skill_id: int, skill_data: dict, db: Session = Depends(get_db)):
    db_skill = db.query(Skill).filter(Skill.id == skill_id).first()
    if not db_skill:
        raise HTTPException(status_code=404, detail="Skill not found")
    
    for key, value in skill_data.items():
        setattr(db_skill, key, value)
    
    db.commit()
    return db_skill

@app.delete("/api/skills/{skill_id}")
async def delete_skill(skill_id: int, db: Session = Depends(get_db)):
    db_skill = db.query(Skill).filter(Skill.id == skill_id).first()
    if not db_skill:
        raise HTTPException(status_code=404, detail="Skill not found")
    
    db.delete(db_skill)
    db.commit()
    return {"status": "success"}

# Contract CRUD
@app.get("/api/contracts")
async def get_contracts(db: Session = Depends(get_db)):
    return db.query(Contract).all()

@app.post("/api/contracts")
async def create_contract(contract_data: dict, db: Session = Depends(get_db)):
    db_contract = Contract(**contract_data)
    db.add(db_contract)
    db.commit()
    db.refresh(db_contract)
    return db_contract

@app.put("/api/contracts/{contract_id}")
async def update_contract(contract_id: int, contract_data: dict, db: Session = Depends(get_db)):
    db_contract = db.query(Contract).filter(Contract.id == contract_id).first()
    if not db_contract:
        raise HTTPException(status_code=404, detail="Contract not found")
    
    for key, value in contract_data.items():
        if key in ['created_at', 'updated_at']:
            value = datetime.fromisoformat(value)
        setattr(db_contract, key, value)
    
    db.commit()
    return db_contract

@app.delete("/api/contracts/{contract_id}")
async def delete_contract(contract_id: int, db: Session = Depends(get_db)):
    db_contract = db.query(Contract).filter(Contract.id == contract_id).first()
    if not db_contract:
        raise HTTPException(status_code=404, detail="Contract not found")
    
    db.delete(db_contract)
    db.commit()
    return {"status": "success"}

# Settings CRUD
@app.get("/api/settings")
async def get_settings(db: Session = Depends(get_db)):
    return db.query(Setting).all()

@app.post("/api/settings")
async def create_setting(setting_data: dict, db: Session = Depends(get_db)):
    db_setting = Setting(**setting_data)
    db.add(db_setting)
    db.commit()
    db.refresh(db_setting)
    return db_setting

@app.put("/api/settings/{setting_id}")
async def update_setting(setting_id: int, setting_data: dict, db: Session = Depends(get_db)):
    db_setting = db.query(Setting).filter(Setting.id == setting_id).first()
    if not db_setting:
        raise HTTPException(status_code=404, detail="Setting not found")
    
    for key, value in setting_data.items():
        setattr(db_setting, key, value)
    
    db.commit()
    return db_setting

@app.delete("/api/settings/{setting_id}")
async def delete_setting(setting_id: int, db: Session = Depends(get_db)):
    db_setting = db.query(Setting).filter(Setting.id == setting_id).first()
    if not db_setting:
        raise HTTPException(status_code=404, detail="Setting not found")
    
    db.delete(db_setting)
    db.commit()
    return {"status": "success"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

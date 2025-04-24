import ee
import time
import datetime
import threading
import sys

# Default settings
NOTIFICATION_ENABLED = True
USE_ICONS = True

def initialize_ee(project_id=None):
    """Initialize Earth Engine with project ID"""
    try:
        if project_id:
            ee.Initialize(project=project_id)
        else:
            ee.Initialize()
        print("Earth Engine initialized successfully")
        return True
    except Exception as e:
        print(f"Error initializing Earth Engine: {e}")
        return False

def get_tasks_batch(task_ids):
    """Get status for multiple tasks in a batch to improve efficiency"""
    try:
        # Request status for all tasks at once
        all_tasks_status = ee.data.getTaskStatus(task_ids)
        
        # Create a dictionary mapping task_id to status
        task_status_dict = {}
        for status in all_tasks_status:
            if 'id' in status:
                task_status_dict[status['id']] = status
            
        return task_status_dict
    except Exception as e:
        print(f"Error getting task status: {e}")
        return {}

def check_task_status(task_id, task_status_dict=None):
    """Check status and information of a specific task"""
    try:
        # Use provided status from batch if available
        if task_status_dict and task_id in task_status_dict:
            status = task_status_dict[task_id]
        else:
            # Fall back to individual request if needed
            try:
                status_list = ee.data.getTaskStatus(task_id)
                if not status_list or len(status_list) == 0:
                    return {'state': 'ERROR', 'description': f'Task {task_id}', 'error_message': 'Task not found'}
                status = status_list[0]
            except Exception as e:
                return {'state': 'ERROR', 'description': f'Task {task_id}', 'error_message': f'API error: {e}'}
        
        # Extract essential fields
        state = status.get('state', 'UNKNOWN')
        description = status.get('description', 'No name')
        
        # Simplify cancellation state handling
        is_cancelled = state in ['CANCELLED', 'CANCEL_COMPLETED'] 
        is_cancel_requested = 'CANCEL' in state or status.get('cancel_requested', False)
        
        result = {
            'state': state,
            'description': description,
            'is_cancelled': is_cancelled,
            'is_cancel_requested': is_cancel_requested
        }
        
        # Add completion time only if finished
        if (state == 'COMPLETED' or is_cancelled) and 'update_timestamp_ms' in status and 'start_timestamp_ms' in status:
            try:
                elapsed_seconds = (int(status['update_timestamp_ms']) - int(status['start_timestamp_ms'])) / 1000
                result['completion_time'] = format_time_elapsed_short(elapsed_seconds)
            except (ValueError, TypeError):
                pass
        
        # Add error message only if failed
        if state == 'FAILED' and 'error_message' in status:
            result['error_message'] = status.get('error_message', 'No error message')
        
        return result
        
    except Exception as e:
        return {'state': 'ERROR', 'description': f'Task {task_id}', 'error_message': f'Unexpected error: {e}'}

def format_time_elapsed_short(seconds):
    """Format runtime to XhXmXs"""
    hours, remainder = divmod(int(seconds), 3600)
    minutes, seconds = divmod(remainder, 60)
    
    if hours > 0:
        return f"{hours}h{minutes}m{seconds}s"
    elif minutes > 0:
        return f"{minutes}m{seconds}s"
    else:
        return f"{seconds}s"

def get_status_icon(state, is_cancelled=False, is_cancel_requested=False):
    """Return icon appropriate to status"""
    if not USE_ICONS:
        return ""
    
    if state == 'ERROR':
        return "⚠️ "
    elif is_cancelled:
        return "🚫 "
    elif is_cancel_requested:
        return "⏹️ "
    elif state == 'COMPLETED':
        return "✅ "
    elif state == 'FAILED':
        return "❌ "
    elif state == 'RUNNING':
        return "⏳ "
    elif state == 'READY':
        return "⏱️ "
    else:
        return "🔄 "

def monitor_tasks(task_ids, interval=10):
    """Monitor multiple tasks with update interval
    
    Args:
        task_ids (list): List of task IDs to monitor
        interval (int): Time between updates (seconds)
    
    Returns:
        bool: True if all completed, False if cancelled
    """
    if not task_ids:
        print("No tasks to monitor")
        return False
    
    # Remove duplicate IDs if any
    task_ids = list(set(task_ids))
    
    all_completed = False
    start_time = time.time()
    total_tasks = len(task_ids)
    
    try:
        while not all_completed:
            all_completed = True
            completed_count = 0
            elapsed_time = time.time() - start_time
            time_str = format_time_elapsed_short(elapsed_time)
            
            # Get status information for all tasks in batch
            task_status_dict = get_tasks_batch(task_ids) or {}
            
            # Display overview status line
            print(f"\n===== Status update ({datetime.datetime.now().strftime('%H:%M:%S')}) | Total time: {time_str} =====")
            
            # Process each task
            task_statuses = []
            for task_id in task_ids:
                # Get status
                status_info = check_task_status(task_id, task_status_dict)
                task_statuses.append(status_info)
                
                # Check if task is complete
                state = status_info['state']
                if state == 'COMPLETED' or state == 'FAILED' or status_info['is_cancelled']:
                    completed_count += 1
                else:
                    all_completed = False
            
            print(f"Progress: {completed_count}/{total_tasks} tasks completed")
            
            # Display detailed information for each task
            for idx, status_info in enumerate(task_statuses, 1):
                state = status_info['state']
                is_cancelled = status_info['is_cancelled']
                is_cancel_requested = status_info['is_cancel_requested']
                description = status_info['description']
                icon = get_status_icon(state, is_cancelled, is_cancel_requested)
                
                # Format status message
                status_text = f"Task {idx}: [{icon}{state}]"
                if 'completion_time' in status_info:
                    status_text += f" [{status_info['completion_time']}]"
                status_text += f" {description}"
                
                # Add error message if present
                if 'error_message' in status_info:
                    status_text += f" - {status_info['error_message']}"
                
                print(status_text)
            
            # Continue if not all tasks are complete
            if not all_completed:
                print(f"\nChecking again in {interval} seconds...")
                print("(Press Ctrl+C to stop monitoring. Tasks will continue running in the background)")
                time.sleep(interval)
        
        # Calculate total time when all completed
        total_elapsed = time.time() - start_time
        total_time_str = format_time_elapsed_short(total_elapsed)
        print(f"\nAll {total_tasks} tasks have completed! Total time: {total_time_str}")
        
        return True
    
    except KeyboardInterrupt:
        interrupted_elapsed = time.time() - start_time
        interrupted_time_str = format_time_elapsed_short(interrupted_elapsed)
        print(f"\nMonitoring stopped after {interrupted_time_str}. Tasks will continue running in the background.")
        print("You can check status at: https://code.earthengine.google.com/tasks")
        return False

def start_export_task(task):
    """Start task and return ID"""
    task.start()
    task_id = task.id
    print(f"Starting task: {task.config['description']} (ID: {task_id})")
    return task_id

def get_recent_tasks(limit=10):
    """Get list of recent tasks"""
    try:
        tasks = ee.data.getTaskList()
        return tasks[:limit]
    except Exception as e:
        print(f"Error when getting task list: {e}")
        return []

def print_task_list(tasks):
    """Print list of tasks"""
    if not tasks:
        print("No tasks to display")
        return
    
    print("\nList of tasks:")
    for i, task in enumerate(tasks, 1):
        state = task['state']
        icon = get_status_icon(state)
        print(f"{i}. {icon}ID: {task['id']}, Name: {task.get('description', 'No name')}, Status: {state}")

def monitor_external_tasks(task_ids, project_id=None):
    """Monitor tasks from external source
    
    Args:
        task_ids (list): List of task IDs to monitor
        project_id (str, optional): Earth Engine Project ID
    """
    success = initialize_ee(project_id)
    if not success:
        print("Failed to initialize Earth Engine. Please check your credentials and try again.")
        return False
    
    if not task_ids:
        print("No tasks to monitor")
        return False
    
    return monitor_tasks(task_ids) 
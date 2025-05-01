// save_schedule.js
// Utility to save the generated schedule to the backend

async function saveGeneratedSchedule(schedule, startDate, duration) {
  try {
    const response = await fetch('/api/save-schedule', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        schedule,
        startDate,
        duration,
      }),
    });
    if (!response.ok) {
      throw new Error('Failed to save schedule');
    }
    return await response.json();
  } catch (error) {
    console.error('Error saving schedule:', error);
    throw error;
  }
}

export { saveGeneratedSchedule };

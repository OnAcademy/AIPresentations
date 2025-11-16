"""
Workflow Automation: Webhook Integration with AI
Building custom workflows that integrate with external services
"""

import json
import hashlib
import hmac
import logging
from typing import Dict, List, Optional
from datetime import datetime
from enum import Enum


# ============================================================================
# EXAMPLE 1: WEBHOOK RECEIVER
# ============================================================================
class WebhookReceiver:
    """
    Receive webhooks from external services (Zapier, external apps)
    """
    
    def __init__(self, secret_key: Optional[str] = None):
        self.secret_key = secret_key
        self.setup_logging()
    
    def setup_logging(self):
        """Setup logging for webhook events"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
    
    def verify_signature(self, payload: str, signature: str) -> bool:
        """
        Verify webhook signature for security
        (Example: Zapier style verification)
        """
        if not self.secret_key:
            return True
        
        expected = hmac.new(
            self.secret_key.encode(),
            payload.encode(),
            hashlib.sha256
        ).hexdigest()
        
        return signature == expected
    
    def handle_webhook(self, payload: Dict) -> Dict:
        """
        Handle incoming webhook data
        """
        try:
            self.logger.info(f"Received webhook: {payload}")
            
            # Process based on event type
            event_type = payload.get('event_type', 'unknown')
            
            if event_type == 'email_received':
                return self.process_email(payload)
            elif event_type == 'form_submitted':
                return self.process_form(payload)
            elif event_type == 'purchase_made':
                return self.process_purchase(payload)
            else:
                return {'status': 'unknown event', 'event_type': event_type}
        
        except Exception as e:
            self.logger.error(f"Error handling webhook: {e}")
            return {'status': 'error', 'message': str(e)}
    
    def process_email(self, payload: Dict) -> Dict:
        """Process email webhook"""
        sender = payload.get('sender')
        subject = payload.get('subject')
        content = payload.get('content')
        
        self.logger.info(f"Processing email from {sender}: {subject}")
        
        return {
            'status': 'processed',
            'type': 'email',
            'sender': sender,
            'subject': subject,
            'has_content': len(content) > 0
        }
    
    def process_form(self, payload: Dict) -> Dict:
        """Process form submission webhook"""
        form_id = payload.get('form_id')
        responses = payload.get('responses', {})
        
        self.logger.info(f"Processing form {form_id} with {len(responses)} responses")
        
        return {
            'status': 'processed',
            'type': 'form',
            'form_id': form_id,
            'response_count': len(responses)
        }
    
    def process_purchase(self, payload: Dict) -> Dict:
        """Process purchase webhook"""
        order_id = payload.get('order_id')
        amount = payload.get('amount')
        customer = payload.get('customer')
        
        self.logger.info(f"Processing purchase {order_id} for {customer}")
        
        return {
            'status': 'processed',
            'type': 'purchase',
            'order_id': order_id,
            'amount': amount
        }


# ============================================================================
# EXAMPLE 2: WORKFLOW ENGINE
# ============================================================================
class WorkflowAction(Enum):
    """Available workflow actions"""
    SEND_EMAIL = "send_email"
    SAVE_TO_DATABASE = "save_to_database"
    CALL_API = "call_api"
    TRANSFORM_DATA = "transform_data"
    CLASSIFY_WITH_AI = "classify_with_ai"
    NOTIFY = "notify"


class Workflow:
    """
    Define and execute automated workflows
    """
    
    def __init__(self, workflow_id: str, trigger: str):
        self.workflow_id = workflow_id
        self.trigger = trigger
        self.steps: List[Dict] = []
        self.logger = logging.getLogger(__name__)
    
    def add_step(self, action: WorkflowAction, config: Dict):
        """Add a step to the workflow"""
        step = {
            'action': action,
            'config': config,
            'order': len(self.steps) + 1
        }
        self.steps.append(step)
        return self
    
    def execute(self, input_data: Dict) -> Dict:
        """Execute workflow steps sequentially"""
        self.logger.info(f"Executing workflow {self.workflow_id}")
        
        context = {'input': input_data, 'output': {}}
        
        for step in self.steps:
            try:
                action = step['action']
                config = step['config']
                
                if action == WorkflowAction.SEND_EMAIL:
                    result = self._send_email(input_data, config)
                
                elif action == WorkflowAction.SAVE_TO_DATABASE:
                    result = self._save_to_database(input_data, config)
                
                elif action == WorkflowAction.TRANSFORM_DATA:
                    result = self._transform_data(input_data, config)
                
                elif action == WorkflowAction.CLASSIFY_WITH_AI:
                    result = self._classify_with_ai(input_data, config)
                
                elif action == WorkflowAction.NOTIFY:
                    result = self._notify(input_data, config)
                
                else:
                    result = {'status': 'unknown action'}
                
                context['output'][f'step_{step["order"]}'] = result
                self.logger.info(f"Step {step['order']}: {result.get('status', 'unknown')}")
                
            except Exception as e:
                self.logger.error(f"Error in step {step['order']}: {e}")
                context['error'] = str(e)
                break
        
        return context
    
    def _send_email(self, data: Dict, config: Dict) -> Dict:
        """Send email action"""
        to = config.get('to')
        subject = config.get('subject')
        body = config.get('body')
        
        self.logger.info(f"Sending email to {to}: {subject}")
        
        # Simulate sending
        return {
            'status': 'email_sent',
            'to': to,
            'subject': subject
        }
    
    def _save_to_database(self, data: Dict, config: Dict) -> Dict:
        """Save to database action"""
        table = config.get('table')
        record = {k: data.get(k) for k in config.get('fields', [])}
        
        self.logger.info(f"Saving to {table}: {record}")
        
        # Simulate saving
        return {
            'status': 'saved',
            'table': table,
            'record_id': 'ID_' + hashlib.md5(str(record).encode()).hexdigest()[:8]
        }
    
    def _transform_data(self, data: Dict, config: Dict) -> Dict:
        """Transform data action"""
        transformations = config.get('transformations', {})
        transformed = data.copy()
        
        for field, transform_func in transformations.items():
            if field in transformed:
                if transform_func == 'uppercase':
                    transformed[field] = transformed[field].upper()
                elif transform_func == 'lowercase':
                    transformed[field] = transformed[field].lower()
        
        self.logger.info(f"Data transformed: {list(transformations.keys())}")
        
        return {
            'status': 'transformed',
            'fields_transformed': list(transformations.keys()),
            'data': transformed
        }
    
    def _classify_with_ai(self, data: Dict, config: Dict) -> Dict:
        """Classify data using AI"""
        field_to_classify = config.get('field')
        categories = config.get('categories', [])
        
        value = data.get(field_to_classify, '')
        
        # Simulate AI classification (in real scenario, call OpenAI)
        classified = self._simulate_classification(value, categories)
        
        self.logger.info(f"Classified '{value[:50]}' as '{classified}'")
        
        return {
            'status': 'classified',
            'field': field_to_classify,
            'original_value': value[:50],
            'classification': classified
        }
    
    def _simulate_classification(self, text: str, categories: List[str]) -> str:
        """Simulate AI classification (replace with real API call)"""
        # In production, call ChatGPT or similar
        # For now, simple keyword matching
        text_lower = text.lower()
        
        for category in categories:
            if category.lower() in text_lower:
                return category
        
        return categories[0] if categories else 'unknown'
    
    def _notify(self, data: Dict, config: Dict) -> Dict:
        """Send notification"""
        channel = config.get('channel')  # 'slack', 'email', 'sms'
        message = config.get('message')
        
        self.logger.info(f"Notification sent to {channel}: {message}")
        
        return {
            'status': 'notified',
            'channel': channel,
            'message': message
        }


# ============================================================================
# EXAMPLE 3: WORKFLOW EXAMPLES
# ============================================================================
def example_customer_support_workflow():
    """
    Workflow: Customer Support Automation
    Trigger: Email received
    Steps: Parse → Classify → Route → Notify
    """
    print("\n" + "=" * 80)
    print("EXAMPLE 1: Customer Support Automation")
    print("=" * 80)
    
    # Create workflow
    workflow = Workflow(
        workflow_id='customer_support_001',
        trigger='email_received'
    )
    
    # Step 1: Transform data (extract fields)
    workflow.add_step(WorkflowAction.TRANSFORM_DATA, {
        'transformations': {
            'sender': 'lowercase',
            'subject': 'uppercase'
        }
    })
    
    # Step 2: Classify urgency with AI
    workflow.add_step(WorkflowAction.CLASSIFY_WITH_AI, {
        'field': 'subject',
        'categories': ['urgent', 'normal', 'low']
    })
    
    # Step 3: Save to database
    workflow.add_step(WorkflowAction.SAVE_TO_DATABASE, {
        'table': 'support_tickets',
        'fields': ['sender', 'subject', 'content']
    })
    
    # Step 4: Send notification
    workflow.add_step(WorkflowAction.NOTIFY, {
        'channel': 'slack',
        'message': 'New support ticket received'
    })
    
    # Execute with sample data
    sample_email = {
        'sender': 'customer@example.com',
        'subject': 'Urgent: Account Login Issues',
        'content': 'Cannot login to my account for 2 days. Please help!'
    }
    
    result = workflow.execute(sample_email)
    
    print("\nWorkflow Execution Result:")
    print(json.dumps(result, indent=2, default=str))


def example_lead_qualification_workflow():
    """
    Workflow: Lead Qualification
    Trigger: Form submitted
    Steps: Enrich → Score → Classify → Route
    """
    print("\n" + "=" * 80)
    print("EXAMPLE 2: Lead Qualification Workflow")
    print("=" * 80)
    
    workflow = Workflow(
        workflow_id='lead_qualification_001',
        trigger='form_submitted'
    )
    
    # Step 1: Transform (standardize data)
    workflow.add_step(WorkflowAction.TRANSFORM_DATA, {
        'transformations': {
            'company_name': 'uppercase',
            'email': 'lowercase'
        }
    })
    
    # Step 2: Classify lead quality
    workflow.add_step(WorkflowAction.CLASSIFY_WITH_AI, {
        'field': 'company_name',
        'categories': ['enterprise', 'mid-market', 'small-business']
    })
    
    # Step 3: Save to CRM
    workflow.add_step(WorkflowAction.SAVE_TO_DATABASE, {
        'table': 'crm_leads',
        'fields': ['name', 'email', 'company_name']
    })
    
    # Step 4: Notify sales team
    workflow.add_step(WorkflowAction.NOTIFY, {
        'channel': 'slack',
        'message': 'High-quality lead received from enterprise'
    })
    
    # Execute with sample data
    sample_lead = {
        'name': 'john doe',
        'email': 'JOHN@ACME.COM',
        'company_name': 'acme corporation',
        'industry': 'technology'
    }
    
    result = workflow.execute(sample_lead)
    
    print("\nWorkflow Execution Result:")
    print(json.dumps(result, indent=2, default=str))


# ============================================================================
# EXAMPLE 4: WORKFLOW STATISTICS
# ============================================================================
class WorkflowStats:
    """Track workflow statistics"""
    
    def __init__(self):
        self.executions = []
    
    def record_execution(self, workflow_id: str, success: bool, duration: float):
        """Record execution statistics"""
        self.executions.append({
            'workflow_id': workflow_id,
            'success': success,
            'duration': duration,
            'timestamp': datetime.now().isoformat()
        })
    
    def get_stats(self) -> Dict:
        """Get aggregated statistics"""
        if not self.executions:
            return {'total': 0, 'success_rate': 0}
        
        total = len(self.executions)
        successful = sum(1 for e in self.executions if e['success'])
        
        return {
            'total_executions': total,
            'successful': successful,
            'failed': total - successful,
            'success_rate': f"{(successful / total * 100):.1f}%",
            'avg_duration': f"{sum(e['duration'] for e in self.executions) / total:.2f}s"
        }
    
    def print_report(self):
        """Print statistics report"""
        stats = self.get_stats()
        
        print("\n" + "=" * 80)
        print("WORKFLOW STATISTICS")
        print("=" * 80)
        
        for key, value in stats.items():
            print(f"{key:<20} {value}")


# ============================================================================
# MAIN EXECUTION
# ============================================================================
if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("WORKFLOW AUTOMATION EXAMPLES")
    print("=" * 80)
    
    # Run examples
    example_customer_support_workflow()
    example_lead_qualification_workflow()
    
    # Show workflow statistics
    stats = WorkflowStats()
    stats.record_execution('customer_support_001', True, 1.23)
    stats.record_execution('customer_support_001', True, 1.15)
    stats.record_execution('lead_qualification_001', True, 0.89)
    stats.print_report()
    
    print("\n" + "=" * 80)
    print("KEY TAKEAWAYS")
    print("=" * 80)
    print("""
1. Workflows automate repetitive multi-step processes
2. Use triggers to start workflows automatically
3. Chain actions together for complex logic
4. Always include error handling
5. Monitor workflow performance
6. Test thoroughly before production
7. Log all executions for debugging
8. Use webhooks for real-time integration

NEXT STEPS:
- Choose automation platform (Zapier, Make, etc.)
- Map out your workflow steps
- Test with real data
- Monitor and optimize
    """)

